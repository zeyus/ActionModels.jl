



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
# - the action probs become NaN when using inital params from the MAP



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
    #subset the ahndata to have two subjID in each clinical_group
    ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344",], ahn_data)
end



#CREATE AGENT
if true

    #Simple model
    function categorical_random(agent::Agent, input::Tuple{Int64, Float64})

        deck, reward = input

        inv_temperature = agent.parameters["inv_temperature"]

        #Do a softmax of the values
        action_probabilities = softmax([0.7, 0.1, 0.1, 0.1] * inv_temperature)

        return Categorical(action_probabilities)
    end

    agent = init_agent(categorical_random,
                            parameters = Dict("inv_temperature" => 1))  

    prior = Dict("inv_temperature" => Uniform(0, 5))

elseif true

    # PVL-Delta
    function pvl_delta(agent::Agent, input::Tuple{Int64, Float64})
        deck, reward = input

        learning_rate = agent.parameters["learning_rate"]
        reward_sensitivity = agent.parameters["reward_sensitivity"]
        loss_aversion = agent.parameters["loss_aversion"]
        inv_temperature = agent.parameters["inv_temperature"]

        expected_value = agent.states["expected_value"]

        action_probabilities = softmax(ActionModels.ad_val.(expected_value) * ActionModels.ad_val(inv_temperature))

        if reward >= 0
            prediction_error = (reward ^ reward_sensitivity) - expected_value[deck]
        else
            prediction_error = -loss_aversion * (abs(reward) ^ reward_sensitivity) - expected_value[deck]
        end

        expected_value[deck] = expected_value[deck] + learning_rate * prediction_error
        
        update_states!(agent, "expected_value", expected_value)

        return Categorical(action_probabilities)
    end

    agent = init_agent(pvl_delta,
                    parameters = Dict(  "learning_rate" => 0.05,
                                        "reward_sensitivity" => 0.4,
                                        "inv_temperature" => 1.3,
                                        "loss_aversion" => 0.5),
                    states = Dict("expected_value" => zeros(Real, 4)
                    ))    

    prior = Dict( 
        "learning_rate"      => LogitNormal(0,1.5),
        "reward_sensitivity" => LogitNormal(0,1.5),
        "inv_temperature"    => Uniform(0,5),
        "loss_aversion"      => Uniform(0,5)
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


# #Simulate data
# if false
#     actions = Vector{Float64}[]
#     learning_rate = "low"
#     for session_input in inputs
#         if learning_rate == "low"

#             set_parameters!(agent, Dict("learning_rate" => 0.05))

#             learning_rate = "high"

#         elseif learning_rate == "high"

#             set_parameters!(agent, Dict("learning_rate" => 0.8))

#             learning_rate = "low"
#         end
#         reset!(agent)
#         session_actions = give_inputs!(agent, session_input)
#         push!(actions, session_actions)
#     end

#     plot(inputs[1]); plot!(actions[1])
#     plot(inputs[4]); plot!(actions[4])
# end

@model function full_model(
    parameter_dists::D,
    parameter_names::Vector{String},
    inputs::Vector{Vector{I}},
    actions::Vector{Vector{A}},
    agents::Vector{Agent},
) where {D<:Distribution,I<:Tuple,A<:Real}

    #Sample parameters
    parameters ~ parameter_dists

    @show parameters

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
# Simple model, subset, real:
#    - NaN's produced when using initial_params
# Simple model, all data, real:

# AD = AutoForwardDiff()                                                                            # works
# AD = AutoReverseDiff(; compile = false)                                                           # works
# AD = AutoReverseDiff(; compile = true)                                                            # works
# import Mooncake; AD = AutoMooncake(; config = nothing);                                           # slow
# import Zygote; using ADTypes: AutoZygote; AD = AutoZygote()                                       # works
# using Enzyme; using ADTypes: AutoEnzyme; AD = AutoEnzyme(; mode=set_runtime_activity(Forward));   # slow

my_sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 20

# result = sample(model, Prior(), n_iterations)

if false
    result = sample(model, my_sampler, n_iterations)

    map_estimate = maximum_a_posteriori(model, adtype=AD)

end

result = sample(model, my_sampler, n_iterations; initial_params=map_estimate.values.array)

result = sample(model, my_sampler, n_iterations; initial_params=manual_parameters)


plot(result)









##### GET THE GRADIENTS / LOGDENSITIES TO SEE IF THEY ARE NaN ######
#Manual parameter settings: learning_rate = 0.001 action_noise = 50
manual_parameters = repeat([0.5], n_agents)





using Turing: LogDensityProblems
ldf = LogDensityFunction(model; adtype=AD)
LogDensityProblems.logdensity(ldf, map_estimate.values.array)
LogDensityProblems.logdensity(ldf, manual_parameters) #using manual parameters
LogDensityProblems.logdensity_and_gradient(ldf, map_estimate.values.array)



# #create a single 


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


###### GETTING INITIAL PARAMETERS FROM PRIOR #######

# using DynamicPPL

# # instantiating a VarInfo samples from the prior; the [:] turns it into a vector of parameter values
# # it would definitely be useful to have a convenience method for this
#        initial_params = DynamicPPL.VarInfo(condmodel)[:]
