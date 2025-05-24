using ActionModels
using CSV, DataFrames

# Example analysis using data from ahn et al 2014
# Iowa Gambling Task on healthy controls and participants with heroin or amphetamine addictions
# More details in docs/example_data/ahn_et_al_2014/ReadMe.txt

####################
### PREPARE DATA ###
####################

# Set path correctly
ActionModels_path = dirname(dirname(pathof(ActionModels)))
docs_path = joinpath(ActionModels_path, "docs")

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

# Make column with total reward
ahn_data[!, :reward] = Float64.(ahn_data[!, :gain] + ahn_data[!, :loss]);

if true
    #subset the ahndata to have two subjID in each clinical_group
    ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344"], ahn_data)
end

action_cols = :deck
observation_cols = (deck = :deck, reward = :reward)
session_cols = :subjID


#################
### PVL-DELTA ###
#################
action_model = ActionModel(PVLDelta(n_decks = 4, act_before_update = true))

model = create_model(
    action_model,
    [
        Regression(@formula(learning_rate ~ clinical_group + (1 | subjID)), logistic),
        Regression(@formula(reward_sensitivity ~ clinical_group + (1 | subjID)), logistic),
        Regression(@formula(loss_aversion ~ clinical_group + (1 | subjID)), exp),
        Regression(@formula(action_precision ~ clinical_group + (1 | subjID)), exp),
    ],
    ahn_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

chains = sample_posterior!(model)

parameters_df = summarize(get_session_parameters!(model))

states_df = get_state_trajectories!(model, :expected_value)









# ### CATEGORICAL RANDOM ####
# function categorical_random(agent::Agent, input::Tuple{Int64,Float64})

#     deck, reward = input

#     action_precision = exp(agent.parameters[:action_precision])

#     #Set the probability 
#     base_probs = [0.1, 0.4, 0.1, 0.1]

#     #Do a softmax of the values
#     action_probabilities = softmax(base_probs * action_precision)

#     return Categorical(action_probabilities)
# end

# action_model = ActionModel(
#     categorical_random,
#     parameters = (; action_precision = Parameter(1)),
#     observations = (deck = Observation(Int64), reward = Observation(Float64)),
#     actions = (; deck = Action(Categorical)),
# )


# model = create_model(
#     action_model,
#     Regression(@formula(action_precision ~ clinical_group + (1 | subjID)), exp),
#     ahn_data,
#     action_cols = action_cols,
#     observation_cols = observation_cols,
#     session_cols = session_cols,
# )

# chains = sample_posterior!(model)