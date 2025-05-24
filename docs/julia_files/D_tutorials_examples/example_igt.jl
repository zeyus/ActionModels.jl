# # Tutorial: Fitting the PVL-Delta model to data from the Iowa Gambling Task
# In this tutorial, we will fit the PVL-Delta model to data from the Iowa Gambling Task (IGT) using the ActionModels.jl package.
# In the IGT, participants choose cards from four decks, each with different reward and loss probabilities, and must learn over time which decks are advantageous.
# We will use data from Ahn et al. (2014), which includes healthy controls and participants with heroin or amphetamine addictions.
# There are more details about the collected data in the docs/example_data/ahn_et_al_2014/ReadMe.txt

using Pkg; Pkg.activate("docs") #Remove later

# ## Loading data
# First, we load the ActionModels package. We also load CSV and Dataframes for loading the data, and StatsPlots for plotting the results.
using ActionModels
using CSV, DataFrames
using StatsPlots

# Then we load the ahn et al. (2014) data, which is available in the docs/example_data/ahn_et_al_2014 folder.

ActionModels_path = dirname(dirname(pathof(ActionModels))) #hide
docs_path = joinpath(ActionModels_path, "docs") #hide

# Import data
data_healthy = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_healthy_control.txt"),
    DataFrame,
)
data_healthy[!, :clinical_group] .= "control"
data_heroin = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_heroin.txt"),
    DataFrame,
)
data_heroin[!, :clinical_group] .= "heroin"
data_amphetamine = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_amphetamine.txt"),
    DataFrame,
)
data_amphetamine[!, :clinical_group] .= "amphetamine"

#Combine into one dataframe
ahn_data = vcat(data_healthy, data_heroin, data_amphetamine)
ahn_data[!, :subjID] = string.(ahn_data[!, :subjID])

# Make column with total reward
ahn_data[!, :reward] = Float64.(ahn_data[!, :gain] + ahn_data[!, :loss]);

show(ahn_data)


# For this example, we will subset the data to only include two subjects from each clinical group.
# This makes the runtime much shorter. Simply skip this step if you want to use the full dataset.

#TODO: Make sure there are also two from the last group
if false
    ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344"], ahn_data)
end

# ## Creating the model
# Then we construct the model to be fitted to the data. 
# We use the PVL-Delta action model, which is a classic model for the IGT.
# In the PVL-Delta is a type of reinfrocement learning model that learns the expected value for each of the decks in the IGT.
# First, the observed reward is transformed with a prospect theory-based utlity curve.
# This means that the subjective value of a reward increses sub-linearly with reward magnitute, and that losses are weighted more heavily than gains.
# The expected value of each deck is then updated using a delta rule, which is the simple reinforcement learning rule used in the classic Rescorla-Wagner model.
# Finally, the probability of selecting each deck is calculated using a softmax function over the expected values of the decks, scaled by an action precision parameter.
# In summary, the PVL-Delta has four parameters: the learning rate $\alpha$, the reward sensitivity $A$, the loss aversion $w$, and the action precision $\beta$.
# See the section REF on the PVL-Delta premade model in the documentation for more details.

# We create the PVL-Delta using the premade model from ActionModels.jl.
# We specify the number of decks, and also that actions are selected before the expected values are updated.
# This is because in the IGT, at least as structured in this dataset, participants select a deck before they receive the reward and update expectations.
action_model = ActionModel(PVLDelta(n_decks = 4, act_before_update = true))

# We then specify whcih column in the data corresponds to the action (deck choice) and which columns correspond to the observations (deck and reward).
# We also specify the columns that uniquely identify each session.
action_cols = :deck
observation_cols = (deck = :deck, reward = :reward)
session_cols = :subjID

# Finally, we create the full model. We use a hierarchical regression model to predict the parameters of the PVL-Delta model based on the clinical group (healthy, heroin, or amphetamine).
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

# ## Fitting the model
# We are now ready to fit the model to the data.

## Set AD backend ##
using ADTypes: AutoReverseDiff, AutoEnzyme
import ReverseDiff
import Enzyme: set_runtime_activity, Reverse 
#ad_type = AutoForwardDiff() # For testing purposes, use ForwardDiff
#ad_type = AutoReverseDiff(; compile = true)
ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true));

## Fit model ##
chains = sample_posterior!(model, n_chains = 1, n_samples = 100, ad_type = ad_type, init_params = nothing)


#TODO: Finish the tutorial

parameters_df = summarize(get_session_parameters!(model))

states_df = summarize(get_state_trajectories!(model, :expected_value))









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