using ActionModels
using Turing, LogExpFunctions
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake
using CSV, DataFrames


###### SETUP ######

action_cols = [:response]
input_cols = [:outcome]
session_cols = [:ID, :session]

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
    groupby(JGET_data, session_cols),
    subdata -> any(ismissing, Matrix(subdata[!, action_cols])) ? DataFrame() : subdata,
)
disallowmissing!(JGET_data, action_cols)
#Remove ID's with missing pdi_scores
JGET_data = combine(
    groupby(JGET_data, session_cols),
    subdata -> any(ismissing, Matrix(subdata[!, [:pdi_total]])) ? DataFrame() : subdata,
)
disallowmissing!(JGET_data, [:pdi_total])


if false
    #subset the data
    JGET_data =
        filter(row -> row[:ID] in [20, 74] && row[:session] in ["1", "2", "3"], JGET_data)
end


### GAUSSIAN RANDOM ###
function gaussian_choice(agent::Agent, input::T) where {T<:Real}

    action_noise = agent.parameters[:action_noise]
    mean = agent.parameters[:mean]

    return Normal(mean, action_noise)
end

action_model = ActionModel(gaussian_choice, parameters = (action_noise = Parameter(1), mean = Parameter(50)))

model = create_model(
    action_model,
    [
        Regression(@formula(mean ~ 1 + session + pdi_total + (1 | ID))),
        Regression(@formula(action_noise ~ 1 + session + pdi_total + (1 | ID)), exp)
    ],
    JGET_data,
    action_cols = action_cols,
    input_cols = input_cols,
    session_cols = session_cols,
)

# AD = AutoForwardDiff()
# AD = AutoReverseDiff(; compile = false)
AD = AutoReverseDiff(; compile = true)
# import Mooncake;
# AD = AutoMooncake(; config = nothing);

#Set samplings settings
sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 10

samples = sample(model, sampler, n_iterations)



### RESCORLA-WAGNER ###
action_model = ActionModel(ContinuousRescorlaWagnerGaussian())

model = create_model(
    action_model,
    [
        Regression(@formula(learning_rate ~ 1 + session + pdi_total + (1 | ID)), logistic),
        Regression(@formula(action_noise ~ 1 + session + pdi_total + (1 | ID)), exp),
        Regression(@formula(initial_value ~ 1 + session + pdi_total + (1 | ID))),
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
    session_cols = session_cols,
)

# AD = AutoForwardDiff()
# AD = AutoReverseDiff(; compile = false)
AD = AutoReverseDiff(; compile = true)
# import Mooncake;
# AD = AutoMooncake(; config = nothing);

#Set samplings settings
sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 10

samples = sample(model, sampler, n_iterations)
