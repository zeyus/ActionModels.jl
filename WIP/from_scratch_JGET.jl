## COMPLEXITY ##
#multiple actions
#dependence on previous action
#updated states
#differentiated array states


## TO DO ##
# The quest:
#   - (DONE!) Move on to the IGT data 
#       - (DONE!) Fix reversediff error
#       - (DONE!) Fix Mooncake error
#   - Move on the HGF
#       - on the JGET data
#       - on the Powers data
#   - Move on to the ActiveInference.jl
#   - Update the package!
#   - Deal with missing data
#   - Deal with multiple inputs 
#   - Deal with multiple actions
#   - Check for type instability
# Make checks:
#   - error on NaN in action cols, regression predictor cols or grouping cols
#   - warning on missing in action cols or regression predictor cols
# Benchmarks:
#   - Use large arraydist for actions
#   - Use multiple filldists for parameters
#   - Go without the Agent struct (send states in and out)
# Implementation:
#   - Use the new version of the full_model
#   - make session_model and population_models modular
#   - use different session_models for missing data or not, and for checking for ParameterRejections
# Issues:
#   - (DONE) Maybe Turing make a warning is there are NaN logdensities? Maybe especially if data being sampled is a NaN? 
#   - (WHATEVER) Maybe Turing can make a warning if the sampler runs into a hard edge of parameter space?
#   - (DONE) Reversediff errors when using a zeros(Real) instead of Zeros(Float64) in PVL-delta
#   - (DONE) The matrix modification in the old version of PVL-delta breaks Turing
#   - (DONE) Mooncake errors when fitting reward sensitivity in PVL-delta




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





grouping_cols = [:ID, :session]
input_cols = [:outcome]
action_cols = [:response]

#Data from https://github.com/nacemikus/jget-schizotypy
#Trial-level data
JGET_data = CSV.read(
    joinpath(docs_path, "example_data/JGET/JGET_data_trial_preprocessed.csv"),
    DataFrame,
    missingstring = ["NaN", ""],
)
JGET_data = select(JGET_data, [:trials, :ID, :session, :outcome, :response, :confidence])

#Subject-level data
subject_data = CSV.read(
    joinpath(docs_path, "example_data/JGET/JGET_data_sub_preprocessed.csv"),
    DataFrame,
    missingstring = ["NaN", ""]
)
subject_data = select(subject_data, [:ID, :session, :pdi_total, :Age, :Gender, :Education])

#Join the data
JGET_data = innerjoin(JGET_data, subject_data, on = [:ID, :session])

#Make session into a categorical variable
JGET_data.session = string.(JGET_data.session)
#Make the outcome column Float64
JGET_data.outcome = Float64.(JGET_data.outcome)

# Remove data groups with missing actions
if true
    grouped_data = groupby(JGET_data, grouping_cols)
    JGET_data = combine(grouped_data, subdata -> any(ismissing, Matrix(subdata[!, action_cols])) ? DataFrame() : subdata)
    disallowmissing!(JGET_data, action_cols)
end

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
                                        "action_noise" => 5),
                    states = Dict("expected_value" => 50)
                    )

    prior = Dict("learning_rate" => LogitNormal(0, 1.5), "action_noise" => Exponential(5))
end

set_save_history!(agent, false)

## THINGS FOR MODEL ##
grouped_data = groupby(JGET_data, grouping_cols)

#Set n_agents
n_agents = length(grouped_data)

#Create agents
agents = [deepcopy(agent) for _ = 1:n_agents];

parameter_dists =
    arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:n_agents]...))


parameter_names = collect(keys(prior))

inputs = [Vector(agent_data[!, first(input_cols)]) for agent_data in grouped_data]

actions = [Vector(agent_data[!, first(action_cols)]) for agent_data in grouped_data]


#Simulate data
if false
    actions = Vector{Float64}[]
    learning_rate = "low"
    for session_input in inputs
        if learning_rate == "low"

            set_parameters!(agent, Dict("learning_rate" => 0.05))

            learning_rate = "high"

        elseif learning_rate == "high"

            set_parameters!(agent, Dict("learning_rate" => 0.8))

            learning_rate = "low"
        end
        reset!(agent)
        session_actions = give_inputs!(agent, session_input)
        push!(actions, session_actions)
    end

    plot(inputs[1]); plot!(actions[1])
    plot(inputs[4]); plot!(actions[4])
end

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

            i ~ to_submodel(single_timestep(agent, input, action))

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

# AD = AutoForwardDiff()                                                                            # works
# AD = AutoReverseDiff(; compile = false)                                                           # works
AD = AutoReverseDiff(; compile = true)                                                            # works
# import Mooncake; AD = AutoMooncake(; config = nothing);                                           # works
# import Zygote; using ADTypes: AutoZygote; AD = AutoZygote()                                       # breaks
# using Enzyme; using ADTypes: AutoEnzyme; AD = AutoEnzyme(; mode=set_runtime_activity(Forward));   # slow

my_sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 1000

# result = sample(model, Prior(), n_iterations)


result = sample(model, my_sampler, n_iterations)

map_estimate = maximum_a_posteriori(model, adtype=AD)
#mle_estimate = maximum_likelihood(model, adtype=AD)

result = sample(model, my_sampler, n_iterations; initial_params=map_estimate.values.array)


plot(density(result[Symbol("parameters[1, 30]")]))

##### GET THE GRADIENTS / LOGDENSITIES TO SEE IF THEY ARE NaN ######
#Manual parameter settings: learning_rate = 0.001 action_noise = 50
manual_parameters = repeat([0.001, 100], n_agents)


using Turing: LogDensityProblems
ldf = LogDensityFunction(model; adtype=AD)
LogDensityProblems.logdensity(ldf, map_estimate.values.array)
LogDensityProblems.logdensity(ldf, manual_parameters) #using manual parameters
LogDensityProblems.logdensity_and_gradient(ldf, map_estimate.values.array)



#create a single 


## CHECK SINGLE PARTICIPANT MODELS ##
#Some participants give NaN logdensities
logdensities = []

for i in 1:n_agents

    parameter_dists_sub = arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:1]...))
    agents_sub = [deepcopy(agent) for _ = 1:1]
    inputs_sub = [inputs[i]]
    actions_sub = [actions[i]]

    model = full_model(parameter_dists_sub, parameter_names, inputs_sub, actions_sub, agents_sub);

    ldf = LogDensityFunction(model; adtype=AD)

    logdensity = LogDensityProblems.logdensity(ldf, map_estimate.values.array)

    push!(logdensities, logdensity)
end

#There are six NaN logdensities
sum(isnan.(logdensities))

#Find the indeces that are NaN
nan_logdensities = findall(isnan, logdensities)

plot(inputs[nan_logdensities[1]]);
plot!(actions[nan_logdensities[1]])

## CHECK MODELS BY PARTICIPANT AMOUNT ##
#NaN logdensities start at 16 participants
logdensities = []

for i in 1:n_agents

    # parameter_dists = arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = 1:i]...))
    # agents = [deepcopy(agent) for _ = 1:i]
    # inputs_sub = inputs[1:i]
    # actions_sub = actions[1:i]

    i = i-1
    parameter_dists_sub = arraydist(hcat([[prior[parameter] for parameter in keys(prior)] for _ = n_agents-i:n_agents]...))
    agents_sub = [deepcopy(agent) for _ = n_agents-i:n_agents]
    inputs_sub = inputs[n_agents-i:n_agents]
    actions_sub = actions[n_agents-i:n_agents]


    model = full_model(parameter_dists_sub, parameter_names, inputs_sub, actions_sub, agents_sub);

    ldf = LogDensityFunction(model; adtype=AD)

    logdensity = LogDensityProblems.logdensity(ldf, map_estimate.values.array)
    #logdensity = LogDensityProblems.logdensity(ldf, manual_parameters)

    push!(logdensities, logdensity)
end

sum(isnan.(logdensities))
logdensities[15]






using Random

@code_warntype model.f(
    model,
    Turing.VarInfo(model),
    Turing.SamplingContext(
        Random.default_rng(), Turing.SampleFromPrior(), Turing.DefaultContext()
    ),
    model.args...,
)






##### ENZYME CAN DO BACKTRACES IF THERE ARE NaN DERIVATIVES ######
# Enzyme.Compiler.CheckNaN = true


###### GETTING INITIAL PARAMETERS FROM PRIOR #######

# using DynamicPPL

# # instantiating a VarInfo samples from the prior; the [:] turns it into a vector of parameter values
# # it would definitely be useful to have a convenience method for this
#        initial_params = DynamicPPL.VarInfo(condmodel)[:]





# sim_actions = give_inputs!(agent, inputs[1])

# using StatsPlots; plot(sim_actions) plot!(inputs[1])
