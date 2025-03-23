docs_path = joinpath(@__DIR__, "..", "..", "docs")
using Pkg
Pkg.activate(docs_path)

using Test
using LogExpFunctions

using ActionModels
using Distributions
using DataFrames, CSV
using Turing
using StatsPlots

# PROBLEMS

# Reversediff errors with PVL-delta (target!)
# no method matching increment_deriv!(::Int64, ::Float64)
# using an array of Real as the base_probs breaks the model!

# Mooncake errors with PVL-delta if reward_sensitivy is estimated 
# - make MWE
# domain error (probably NaN probs)

# Optim gets to -Inf params (Solved)
# - fix: clamp + normalize the probs

#Matrix modification (Solved: workaround)
# - the matrix modification breaks the model
# - results in a weird error with forwarddiff
# - make an MWE
# - workaround: create a new matrix

# NaN params with hard edges, when using MAP (Solved: needs a warning)
# - the action probs become NaN when using inital params from the MAP
# - this is because the probability get's squashed towards the minimum value, so the graident become NaN when the value is exactly on the minimum value
# - Inf lp's with wide priors; it's the optim that leads to weird places
# - make a MWE

#IDEAS:
# - don't use Agent struct (necessary for MWE's too)
# - normalize rewards


action_cols = [:deck]
input_cols = [:deck, :reward]
grouping_cols = [:subjID]



# Import data
data_healthy = CSV.read(joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_healthy_control.txt"), DataFrame)
data_healthy[!, :clinical_group] .= "healthy"
data_heroin = CSV.read(joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_heroin.txt"), DataFrame)
data_heroin[!, :clinical_group] .= "heroin"
data_amphetamine = CSV.read(joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_amphetamine.txt"), DataFrame)
data_amphetamine[!, :clinical_group] .= "amphetamine"

#Combine into one dataframe
ahn_data = vcat(data_healthy, data_heroin, data_amphetamine)
ahn_data[!, :subjID] = string.(ahn_data[!, :subjID])
# Make clumn with total reward
ahn_data[!, :reward] = Float64.(ahn_data[!, :gain] + ahn_data[!, :loss]);

if false
    # #subset the ahndata to have two subjID in each clinical_group
    # ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344",], ahn_data)
    #subset the ahndata to have two subjID in each clinical_group
    ahn_data = filter(row -> row[:subjID] in ["103",], ahn_data)
end



#CREATE AGENT
if false

    #Simple model
    function categorical_random(agent::Agent, input::Tuple{Int64, Float64})

        deck, reward = input

        inv_temperature = exp(agent.parameters["log_inv_temperature"])

        #Set the probability 
        base_probs = [0.1, 0.1, 0.1, 0.1]
        #base_probs = zeros(Real, 4) # - USING THIS BREAKD REVERSEDIFF
        base_probs[1] = 0.7

        #Do a softmax of the values
        action_probabilities = softmax(base_probs * inv_temperature)

        return Categorical(action_probabilities)
    end

    agent = init_agent(categorical_random,
                            parameters = Dict("log_inv_temperature" => 1),
                            states = Dict("expected_value" => zeros(Float64, 4)
                    ))  

    prior = Dict("log_inv_temperature" => Normal(0,2))

elseif true

    # PVL-Delta
    function pvl_delta(agent::Agent, input::Tuple{Int64, Float64})
        deck, reward = input

        learning_rate = logistic(agent.parameters["logit_learning_rate"])
        reward_sensitivity = logistic(agent.parameters["logit_reward_sensitivity"])
        loss_aversion = exp(agent.parameters["log_loss_aversion"])
        inv_temperature = exp(agent.parameters["log_inv_temperature"])

        expected_value = agent.states["expected_value"]

        action_probabilities = softmax(expected_value * inv_temperature)
        
        #Avoid underflow and overflow
        action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
        action_probabilities = action_probabilities / sum(action_probabilities)

        if reward >= 0
            prediction_error = (reward ^ reward_sensitivity) - expected_value[deck]
        else
            prediction_error = -loss_aversion * (abs(reward) ^ reward_sensitivity) - expected_value[deck]
        end

        new_expected_value = [
            expected_value[deck_idx] + learning_rate * prediction_error * (deck == deck_idx) for deck_idx in 1:4 
        ]
        
        update_states!(agent, "expected_value", new_expected_value)

        return Categorical(action_probabilities)
    end

    agent = init_agent(pvl_delta,
                    parameters = Dict(  "logit_learning_rate" => -2,
                                        "logit_reward_sensitivity" => 0,
                                        "log_inv_temperature" => 0,
                                        "log_loss_aversion" => 0),
                    states = Dict("expected_value" => zeros(Float64, 4)
                    ))    

    prior = Dict( 
        "logit_learning_rate"      => Normal(0,1),
        "logit_reward_sensitivity" => Normal(0,1),
        "log_inv_temperature"      => Normal(0,1),
        "log_loss_aversion"        => Normal(0,1)
        )
end

set_save_history!(agent, false)

## THINGS FOR MODEL ##
grouped_data = groupby(ahn_data, grouping_cols)

#Set n_agents
n_agents = length(grouped_data)

#Create agents
agents = [deepcopy(agent) for _ = 1:n_agents];

parameter_dists =
    arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:n_agents]...))


parameter_names = collect(keys(prior))

inputs = [Tuple.(eachrow(agent_data[!, input_cols])) for agent_data in grouped_data]

actions = [first.(eachrow(agent_data[!, first(action_cols)])) for agent_data in grouped_data]



@model function full_model(
    parameter_dists::D,
    parameter_names::Vector{String},
    inputs::Vector{Vector{I}},
    actions::Vector{Vector{A}},
    agents::Vector{Agent},
) where {D<:Distribution,I<:Tuple,A<:Real}

    #Sample parameters
    parameters ~ parameter_dists

    #For each session
    for (session_parameters, session_inputs, session_actions, agent) in
        zip(eachcol(parameters), inputs, actions, agents)

        #Prepare the agent
        set_parameters!(agent, parameter_names, session_parameters)
        reset!(agent)

        #For each timestep
        for (input, action) in zip(session_inputs, session_actions)

            i ~ to_submodel(single_timestep(agent, input, action))

        end
    end
end


@model function single_timestep(agent::Agent, input::I, action::A) where {I<:Tuple,A<:Real}

    #Give input and sample action
    action ~ agent.action_model(agent, input)

    #Store the agent's action in the agent
    update_states!(agent, "action", action)

end


model = full_model(parameter_dists, parameter_names, inputs, actions, agents);


### SAMPLE ###

#CHECK:
# Simple model, all data, real: works, but MAP leads to -Inf (solved by clamping?)
# PVL, subset, real, all params:
#   - forwarddiff: works
#   - reversdiff: no method matching increment_deriv!(::Int64, ::Float64)
#   - mooncake: can't find initial parameters. a Domainerror happens
# PVL, all data, real, all params:
#   - forwarddiff: works (but slow)
#   - reversdiff: 
#   - reversdiff(true): 
#   - mooncake: 
# PVL, all data, real, no reward_sensitivity:
#   - forwarddiff: works 
#   - reversdiff: no method matching increment_deriv!(::Int64, ::Float64)
#   - reversdiff(true): 
#   - mooncake: works
# PVL, subset, simulated, all params:
#   - forwarddiff: works
#   - reversdiff: no method matching increment_deriv!(::Int64, ::Float64)
#   - reversdiff(true): same
#   - mooncake: can't find initial parameters. a Domainerror happens
# PVL, all data, simulated, all params:
#   - forwarddiff: can't find starting point; with initial params, logprob is -Inf, and the action probs are NaN. 
#                  narrow priors avoids this. Expect it to be underflow with categorical dist probs
#   - reversdiff: 
#   - reversdiff(true): 
#   - mooncake: 


# AD = AutoForwardDiff()                                                                            # works
# AD = AutoReverseDiff(; compile = false)                                                           # works
AD = AutoReverseDiff(; compile = true)                                                            # works
# import Mooncake; AD = AutoMooncake(; config = nothing);                                           # slow

my_sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 1500



# result = sample(model, Prior(), n_iterations)

if false

    result = sample(model, my_sampler, n_iterations)

elseif false
    
    map_estimate = maximum_a_posteriori(model, adtype=AD)

    result = sample(model, my_sampler, n_iterations; initial_params=map_estimate.values.array)
    
    plot(result)
end





# result = sample(model, my_sampler, n_iterations; initial_params=prior_params)
   


##### GET THE GRADIENTS / LOGDENSITIES TO SEE IF THEY ARE NaN ######
#Manual parameter settings: learning_rate = 0.001 action_noise = 50
manual_parameters = repeat([0, 0, 0, 0], n_agents)

using Turing: DynamicPPL; prior_params = DynamicPPL.VarInfo(model)[:]

using Turing: LogDensityProblems
ldf = LogDensityFunction(model; adtype=AD)
# LogDensityProblems.logdensity(ldf, map_estimate.values.array)
#LogDensityProblems.logdensity(ldf, manual_parameters) #using manual parameters
LogDensityProblems.logdensity(ldf, prior_params) 

# LogDensityProblems.logdensity_and_gradient(ldf, map_estimate.values.array)
#LogDensityProblems.logdensity_and_gradient(ldf, manual_parameters)
LogDensityProblems.logdensity_and_gradient(ldf, prior_params)




#Test many prior param settings
many_prior_params = [ DynamicPPL.VarInfo(model)[:] for _ = 1:100]
logdensities = [LogDensityProblems.logdensity(ldf, prior_params) for prior_params in many_prior_params]




# ## CHECK SINGLE PARTICIPANT MODELS ##
# #Some participants give NaN logdensities
# logdensities = []

# for i in 1:n_agents

#     parameter_dists_sub = arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:1]...))
#     agents_sub = [deepcopy(agent) for _ = 1:1]
#     inputs_sub = [inputs[i]]
#     actions_sub = [actions[i]]

#     model = full_model(parameter_dists_sub, parameter_names, inputs_sub, actions_sub, agents_sub);

#     ldf = LogDensityFunction(model; adtype=AD)

#     logdensity = LogDensityProblems.logdensity(ldf, map_estimate.values.array)

#     push!(logdensities, logdensity)
# end

# #There are six NaN logdensities
# sum(isnan.(logdensities))

# #Find the indeces that are NaN
# nan_logdensities = findall(isnan, logdensities)

# plot(inputs[nan_logdensities[1]]);
# plot!(actions[nan_logdensities[1]])

# ## CHECK MODELS BY PARTICIPANT AMOUNT ##
# #NaN logdensities start at 16 participants
# logdensities = []

# for i in 1:n_agents

#     # parameter_dists = arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:i]...))
#     # agents = [deepcopy(agent) for _ = 1:i]
#     # inputs_sub = inputs[1:i]
#     # actions_sub = actions[1:i]

#     i = i-1
#     parameter_dists_sub = arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = n_agents-i:n_agents]...))
#     agents_sub = [deepcopy(agent) for _ = n_agents-i:n_agents]
#     inputs_sub = inputs[n_agents-i:n_agents]
#     actions_sub = actions[n_agents-i:n_agents]


#     model = full_model(parameter_dists_sub, parameter_names, inputs_sub, actions_sub, agents_sub);

#     ldf = LogDensityFunction(model; adtype=AD)

#     logdensity = LogDensityProblems.logdensity(ldf, map_estimate.values.array)
#     #logdensity = LogDensityProblems.logdensity(ldf, manual_parameters)

#     push!(logdensities, logdensity)
# end

# sum(isnan.(logdensities))
# logdensities[15]






# using Random

# @code_warntype model.f(
#     model,
#     Turing.VarInfo(model),
#     Turing.SamplingContext(
#         Random.default_rng(), Turing.SampleFromPrior(), Turing.DefaultContext()
#     ),
#     model.args...,
# )






##### ENZYME CAN DO BACKTRACES IF THERE ARE NaN DERIVATIVES ######
# Enzyme.Compiler.CheckNaN = true




### ENZYME AND ZYGOTE ###
# import Zygote; using ADTypes: AutoZygote; AD = AutoZygote()                                       # works
# using Enzyme; using ADTypes: AutoEnzyme; AD = AutoEnzyme(; mode=set_runtime_activity(Forward));   # slow







##### SIMULATE DATA ######
# #Simulate data
# if false
    
#     actions = Vector{Int64}[]
    
    
#     type = 1

#     for session_input in inputs
#         if type == 1

#             session_params = Dict(  
#                 "logit_learning_rate" => -2,
#                 "logit_reward_sensitivity" => 1,
#                 "log_inv_temperature" => -1,
#                 "log_loss_aversion" => -1
#                 )
            

#             type = 2

#         elseif type == 2

#             session_params = Dict(  
#                 "logit_learning_rate" => 2,
#                 "logit_reward_sensitivity" => 1,
#                 "log_inv_temperature" => -1,
#                 "log_loss_aversion" => -1
#                 )

#             type = 3

#         elseif type == 3

#             session_params = Dict(  
#                 "logit_learning_rate" => 0,
#                 "logit_reward_sensitivity" => -1,
#                 "log_inv_temperature" => -1,
#                 "log_loss_aversion" => -1
#                 )

#             type = 1
#         end

#         set_parameters!(agent, session_params)
#         reset!(agent)

#         session_actions = [single_input!(agent, input) for input in session_input]

#         push!(actions, session_actions)
#     end

#     plot(inputs[1]); plot(actions[1])
#     plot(inputs[4]); plot!(actions[2])
# end