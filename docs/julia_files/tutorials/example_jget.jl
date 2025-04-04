docs_path = joinpath(@__DIR__, "..", "..")
using Pkg
Pkg.activate(docs_path)

using ActionModels
using Turing, LogExpFunctions
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake
using CSV, DataFrames


###### SETUP ######

action_cols = [:response]
input_cols = [:outcome]
grouping_cols = [:ID, :session]

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
    missingstring = ["NaN", ""],
)
subject_data = select(subject_data, [:ID, :session, :pdi_total, :Age, :Gender, :Education])

#Join the data
JGET_data = innerjoin(JGET_data, subject_data, on = [:ID, :session])

#Make session into a categorical variable
JGET_data.session = string.(JGET_data.session)
#Make the outcome column Float64
JGET_data.outcome = Float64.(JGET_data.outcome)

#Remove ID's with missing actions
JGET_data = combine(
    groupby(JGET_data, grouping_cols),
    subdata -> any(ismissing, Matrix(subdata[!, action_cols])) ? DataFrame() : subdata,
)
disallowmissing!(JGET_data, action_cols)
#Remove ID's with missing pdi_scores
JGET_data = combine(
    groupby(JGET_data, grouping_cols),
    subdata -> any(ismissing, Matrix(subdata[!, [:pdi_total]])) ? DataFrame() : subdata,
)
disallowmissing!(JGET_data, [:pdi_total])

#Normalise inputs and responses
#TODO: 

if false
    #subset the data
    JGET_data =
        filter(row -> row[:ID] in [20, 74] && row[:session] in ["1", "2", "3"], JGET_data)
end


### GAUSSIAN RANDOM ###
function gaussian_choice(agent::Agent, input::T) where {T<:Real}

    action_noise = agent.parameters["action_noise"]
    mean = agent.parameters["mean"]

    return Normal(mean, action_noise)
end

agent = init_agent(gaussian_choice, parameters = Dict("action_noise" => 1, "mean" => 50))

model = create_model(
    agent,
    [
        @formula(mean ~ 1 + session + pdi_total + (1 | ID)),
        @formula(action_noise ~ 1 + session + pdi_total + (1 | ID))
    ],
    JGET_data,
    priors = [
        RegressionPrior(β = Normal(50, 30), σ = Exponential(1)),
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
    ],
    inv_links = Function[identity, exp],
    action_cols = action_cols,
    input_cols = input_cols,
    grouping_cols = grouping_cols,
)

# AD = AutoForwardDiff()
# AD = AutoReverseDiff(; compile = false)
# AD = AutoReverseDiff(; compile = true)
import Mooncake;
AD = AutoMooncake(; config = nothing);

#Set samplings settings
sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 10

samples = sample(model, sampler, n_iterations)



### RESCORLA-WAGNER ###
function rescorla_wagner(agent::Agent, input::T) where {T<:Real}

    learning_rate = agent.parameters["learning_rate"]
    action_noise = agent.parameters["action_noise"]

    expected_value = agent.states["value"]

    action_probability = Normal(expected_value, action_noise)

    new_expected_value = expected_value + learning_rate * (input - expected_value)

    update_states!(agent, "value", new_expected_value)

    return action_probability
end

agent = init_agent(
    rescorla_wagner,
    parameters = Dict(
        "learning_rate" => 0.2,
        "action_noise" => 5,
        "initial_value" => InitialState("value", 50),
    ),
    states = Dict("value" => 50),
)

model = create_model(
    agent,
    [
        @formula(learning_rate ~ 1 + session + pdi_total + (1 | ID)),
        @formula(action_noise ~ 1 + session + pdi_total + (1 | ID)),
        @formula(initial_value ~ 1 + session + pdi_total + (1 | ID)),
    ],
    JGET_data,
    priors = [
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
        RegressionPrior(),
    ],
    inv_links = Function[logistic, exp, identity],
    action_cols = action_cols,
    input_cols = input_cols,
    grouping_cols = grouping_cols,
)

# AD = AutoForwardDiff()
# AD = AutoReverseDiff(; compile = false)
# AD = AutoReverseDiff(; compile = true)
import Mooncake;
AD = AutoMooncake(; config = nothing);

#Set samplings settings
sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 10

samples = sample(model, sampler, n_iterations)
