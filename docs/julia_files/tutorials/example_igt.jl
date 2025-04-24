using ActionModels
using Turing, LogExpFunctions
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake
using CSV, DataFrames


# Example analysis using data from ahn et al 2014
# Iowa Gambling Task on healthy controls and participants with heroin or amphetamine addictions
# More details in docs/example_data/ahn_et_al_2014/ReadMe.txt

action_cols = [:deck]
input_cols = [:deck, :reward]
grouping_cols = [:subjID]

# Import data
data_healthy = CSV.read(
    joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_healthy_control.txt"),
    DataFrame,
)
data_healthy[!, :clinical_group] .= "healthy"
data_heroin = CSV.read(
    joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_heroin.txt"),
    DataFrame,
)
data_heroin[!, :clinical_group] .= "heroin"
data_amphetamine = CSV.read(
    joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_amphetamine.txt"),
    DataFrame,
)
data_amphetamine[!, :clinical_group] .= "amphetamine"

#Combine into one dataframe
ahn_data = vcat(data_healthy, data_heroin, data_amphetamine)
ahn_data[!, :subjID] = string.(ahn_data[!, :subjID])

# Make column wit total reward
ahn_data[!, :reward] = Float64.(ahn_data[!, :gain] + ahn_data[!, :loss]);


if false
    #subset the ahndata to have two subjID in each clinical_group
    ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344"], ahn_data)
end






### CATEGORICAL RANDOM ####
function categorical_random(agent::Agent, input::Tuple{Int64,Float64})

    deck, reward = input

    inv_temperature = exp(agent.parameters[:inv_temperature])

    #Set the probability 
    base_probs = [0.1, 0.4, 0.1, 0.1]

    #Do a softmax of the values
    action_probabilities = softmax(base_probs * inv_temperature)

    return Categorical(action_probabilities)
end

action_model = ActionModel(
    categorical_random,
    parameters = (; inv_temperature = Parameter(1)),
    observations = (deck = Observation(Int64), reward = Observation(Float64)),
    actions = (; deck = Action(Categorical)),
)


model = create_model(
    action_model,
    @formula(inv_temperature ~ clinical_group + (1 | subjID)),
    ahn_data,
    # priors = RegressionPrior(β = [Normal(0, 0.1)]),
    inv_links = exp,
    action_cols = [:deck],
    input_cols = [:deck, :reward],
    grouping_cols = [:subjID],
)

# AD = AutoForwardDiff()
# AD = AutoReverseDiff(; compile = false)
AD = AutoReverseDiff(; compile = true)
# import Mooncake; AD = AutoMooncake(; config = nothing); 

#Set samplings settings
sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 10

samples = sample(model, sampler, n_iterations)


### PVL-DELTA ###
action_model = ActionModel(PVLDelta(n_decks = 4))

model = create_model(
    action_model,
    [
        @formula(learning_rate ~ clinical_group + (1 | subjID)),
        @formula(reward_sensitivity ~ clinical_group + (1 | subjID)),
        @formula(loss_aversion ~ clinical_group + (1 | subjID)),
        @formula(inv_temperature ~ clinical_group + (1 | subjID)),
    ],
    ahn_data,
    priors = [
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
        RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
    ],
    inv_links = [logistic, logistic, exp, exp],
    action_cols = action_cols,
    input_cols = input_cols,
    grouping_cols = grouping_cols,
)

# AD = AutoForwardDiff()
# AD = AutoReverseDiff(; compile = false)
AD = AutoReverseDiff(; compile = true) 
# import Mooncake; AD = AutoMooncake(; config = nothing);

#Set samplings settings
sampler = NUTS(-1, 0.65; adtype = AD)
n_iterations = 10

samples = sample(model, sampler, n_iterations)