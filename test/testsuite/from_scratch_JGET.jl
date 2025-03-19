### VARIATION ###
#Don't rely on agent struct
#   - send new states in and out of function
#Use multiple filldist instead of one arraydist
#Use one big arraydist for actions
#Use more advanced agent (rescorla)
#Figure out how to make enzyme compatible

## COMPLEXITY ##
#multiple actions
#dependence on previous action
#updated states
#differentiated array states

## PROBLEMS ##
#Update_states! leads to NaN log-probabilities
#   - change to not use Agent struct!

docs_path = joinpath(@__DIR__, "..", "..", "docs")
using Pkg
Pkg.activate(docs_path)

using Test
using LogExpFunctions

using ActionModels
using Distributions
using DataFrames, CSV
using MixedModels
using Turing




#Data from https://github.com/nacemikus/jget-schizotypy
#Trial-level data
JGET_data = CSV.read(
    joinpath(docs_path, "example_data/JGET/JGET_data_trial_preprocessed.csv"),
    DataFrame,
)
JGET_data = select(JGET_data, [:trials, :ID, :session, :outcome, :response, :confidence])

#Subject-level data
subject_data = CSV.read(
    joinpath(docs_path, "example_data/JGET/JGET_data_sub_preprocessed.csv"),
    DataFrame,
)
subject_data = select(subject_data, [:ID, :session, :pdi_total, :Age, :Gender, :Education])

#Join the data
JGET_data = innerjoin(JGET_data, subject_data, on = [:ID, :session])

#Remove rows with missing values in pdi_total
JGET_data = filter(row -> !ismissing(row[:pdi_total]), JGET_data)
#Make session into a categorical variable
JGET_data.session = string.(JGET_data.session)
#Make the outcome column Float64
JGET_data.outcome = Float64.(JGET_data.outcome)

if true
    #subset the data
    JGET_data =
        filter(row -> row[:ID] in [20, 74] && row[:session] in ["1", "2", "3"], JGET_data)
end

#CREATE AGENT
if false

    function gaussian_choice(agent::Agent, input::T) where {T<:Real}
        action_noise = agent.parameters["action_noise"]
        mean = agent.parameters["mean"]

        return Normal(mean, action_noise)
    end

    agent =
        init_agent(gaussian_choice, parameters = Dict("action_noise" => 1, "mean" => 50))

    prior = Dict("action_noise" => LogNormal(0, 1), "mean" => Normal(50, 20))

elseif true

    # create resocrla wagner aget
    function rescorla_wagner(agent::Agent, input::T) where T<:Real

        learning_rate = agent.parameters["learning_rate"]
        action_noise = agent.parameters["action_noise"]

        expected_value = agent.states["expected_value"]

        action_probability = Normal(expected_value, action_noise)

        new_expected_value = expected_value + learning_rate * (input - expected_value)
        
        update_states!(agent, "expected_value", new_expected_value)

        return action_probability
    end

    agent = init_agent(rescorla_wagner,
                    parameters = Dict(  "learning_rate" => 0.2,
                                        "action_noise" => 20),
                    states = Dict("expected_value" => 50)
                    )

    prior = Dict("learning_rate" => LogitNormal(0, 1), "action_noise" => Exponential(15))
end

set_save_history!(agent, false)

grouping_cols = [:ID, :session]
input_cols = [:outcome]
action_cols = [:response]


## THINGS FOR MODEL ##
grouped_data = groupby(JGET_data, grouping_cols)
n_agents = length(grouped_data)

parameter_dists =
    arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:n_agents]...))

parameter_names = collect(keys(prior))

inputs = [Vector(agent_data[!, first(input_cols)]) for agent_data in grouped_data]

actions = [Vector(agent_data[!, first(action_cols)]) for agent_data in grouped_data]

actions = []
for session_input in inputs
    reset!(agent)
    session_actions = give_inputs!(agent, session_input)
    push!(actions, session_actions)
end

#agents = [agent for _ = 1:n_agents];
agents = [deepcopy(agent) for _ = 1:n_agents];

@model function full_model(
    parameter_dists::D,
    parameter_names::Vector{String},
    inputs::Vector{Vector{I}},
    actions::Vector{Vector{A}},
    agents::Vector{Agent},
) where {D<:Distribution,I<:Real,A<:Real}

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

            @submodel single_timestep(agent, input, action)

        end
    end
end

@model function single_timestep(agent::Agent, input::I, action::A) where {I<:Real,A<:Real}

    #Give input and sample action
    action ~ agent.action_model(agent, input)

    #Store the agent's action in the agent
    update_states!(agent, "action", action)

end


model = full_model(parameter_dists, parameter_names, inputs, actions, agents);


### SAMPLE ###

# Simple model, full data: ADs all work
# RW, subset data: breaks
# RW, full data: 

AD = AutoForwardDiff()                                                        # works
# import ReverseDiff; AD = AutoReverseDiff(; compile = false)                                       # works
# import ReverseDiff; AD = AutoReverseDiff(; compile = true)                                        # works
# import Mooncake; AD = AutoMooncake(; config = nothing);                                           # works
# import Zygote; AD = AutoZygote()                                                                  # works
# using Enzyme; using ADTypes: AutoEnzyme; AD = AutoEnzyme(; mode=set_runtime_activity(Forward));   # works

my_sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 1000

# result = sample(model, Prior(), n_iterations)


result = sample(model, my_sampler, n_iterations)

map_estimate = maximum_a_posteriori(model, adtype=AD)
mle_estimate = maximum_likelihood(model, adtype=AD)

result = sample(model, my_sampler, n_iterations; initial_params=map_estimate.values.array)


using StatsPlots;
plot(result);





##### GET THE GRADIENTS TO SEE IF THEY ARE NaN ######

# using DynamicPPL: @model, LogDensityFunction
# using Distributions
# using LogDensityProblems: logdensity_and_gradient
# using LogDensityProblemsAD: ADgradient


# import ForwardDiff
# ℓ = ADgradient(:ForwardDiff, LogDensityFunction(model1()))
# logdensity_and_gradient(ℓ, [1.0, 2.0])
# # --> (-3.0285667753085077, [NaN, NaN])

# import Mooncake
# ℓ = ADgradient(:Mooncake, LogDensityFunction(model1()))
# logdensity_and_gradient(ℓ, [1.0, 2.0, 3.0])
# # (-3.0285667753085077, [NaN, -2.0])

# import ReverseDiff
# ℓ = ADgradient(:ReverseDiff, LogDensityFunction(model1()))
# logdensity_and_gradient(ℓ, [1.0, 2.0, 3.0])
# # (-3.0285667753085077, [NaN, -2.0])

# import Zygote
# ℓ = ADgradient(:Zygote, LogDensityFunction(model1()))
# logdensity_and_gradient(ℓ, [1.0, 2.0, 3.0])
# # (-3.0285667753085077, [NaN, -2.0])





##### ENZYME CAN DO BACKTRACES IF THERE ARE NaN DERIVATIVES ######
# Enzyme.Compiler.CheckNaN = true


###### GETTING INITIAL PARAMETERS FROM PRIOR #######

# using DynamicPPL

# # instantiating a VarInfo samples from the prior; the [:] turns it into a vector of parameter values
# # it would definitely be useful to have a convenience method for this
#        initial_params = DynamicPPL.VarInfo(condmodel)[:]





# sim_actions = give_inputs!(agent, inputs[1])

# using StatsPlots; plot(sim_actions) plot!(inputs[1])
