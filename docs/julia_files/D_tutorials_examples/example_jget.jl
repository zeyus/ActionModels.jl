# # Tutorial: Fitting data from the Jumping Gaussian Estimation Task
# In this tutorial, we will fit a Rescorla-Wagner model to data from the Jumping Gaussian Estimation Task (JGET).
# In the JGET, participants observe continuous outomces sampled from a Gaussian distribution which at some trials "jumps" to be centered somewhere else.
# Participants must predict the outcome of the next trial based on the previous outcomes.
# The data we will use is from a study on schizotypy, where participants completed the JGET and also filled out the Peters Delusions Inventory (PDI).
# The PDI is a self-report questionnaire that measures delusional ideation, and we will use it as a predictor in our model.
# Data is taken from https://github.com/nacemikus/jget-schizotypy, where more information can also be found.

using Pkg;
Pkg.activate("docs") #Remove later

# ## Loading data
# First, we load the ActionModels package. We also load CSV and Dataframes for loading the data, and StatsPlots for plotting the results.
using ActionModels
using CSV, DataFrames
using StatsPlots

# Then we load the JGET data, which is available in the docs/example_data/JGET folder.

ActionModels_path = dirname(dirname(pathof(ActionModels))) #hide
docs_path = joinpath(ActionModels_path, "docs") #hide

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
#Make the outcome into a Float64
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


show(JGET_data)



# For this example, we will subset the data to only include two subjects in total, across the three epxerimental sessions.
# This makes the runtime much shorter. Simply skip this step if you want to use the full dataset.

if false
    JGET_data = filter(row -> row[:ID] in [20, 74], JGET_data)
end




# ## Creating the model
# Then we construct the model to be fitted to the data.
# We will use a classic Rescorla-Wagner model with a Gaussian report action model.
# The Rescorla-Wagner model is a simple reinforcement learning model that updates the expected value of an action based on the observed outcome.
# The Gaussian report action model assumes that the agent reports a continuous value sampled from a Gaussian distribution, where the mean is the expected value of the action and the standard deviation is a noise parameter.
# There are then two parameters in the action model: the learning rate $\alpha$ and the action noise $\beta$.

# We create the Rescorla-Wagner action model using the premade model from ActionModels.jl.
action_model = ActionModel(PremadeRescorlaWagner())

# We then specify which column in the data corresponds to the action (the response) and which columns correspond to the input (the outcome).
# There are twp columns which jointly specify the sessions: the ID of the participant and the experimental session number.
action_cols = :response
input_cols = :outcome
session_cols = [:ID, :session]

# We then create the fulle model. We use a hierarchical regression model to predict the parameters of the Rescorla-Wagner model based on the session number and the PDI score.
model = create_model(
    action_model,
    [
        Regression(@formula(learning_rate ~ 1 + session + pdi_total + (1 | ID)), logistic),
        Regression(@formula(action_noise ~ 1 + session + pdi_total + (1 | ID)), exp),
    ],
    JGET_data,
    action_cols = action_cols,
    input_cols = input_cols,
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














# ### GAUSSIAN RANDOM ###
# function gaussian_choice(agent::Agent, input::T) where {T<:Real}

#     action_noise = agent.parameters[:action_noise]
#     mean = agent.parameters[:mean]

#     return Normal(mean, action_noise)
# end

# action_model = ActionModel(
#     gaussian_choice,
#     parameters = (action_noise = Parameter(1), mean = Parameter(50)),
# )

# model = create_model(
#     action_model,
#     [
#         Regression(@formula(mean ~ 1 + session + pdi_total + (1 | ID))),
#         Regression(@formula(action_noise ~ 1 + session + pdi_total + (1 | ID)), exp),
#     ],
#     JGET_data,
#     action_cols = action_cols,
#     input_cols = input_cols,
#     session_cols = session_cols,
# )

# # AD = AutoForwardDiff()
# # AD = AutoReverseDiff(; compile = false)
# AD = AutoReverseDiff(; compile = true)
# # import Mooncake;
# # AD = AutoMooncake(; config = nothing);

# #Set samplings settings
# sampler = NUTS(-1, 0.65; adtype = AD)
# n_iterations = 10

# samples = sample(model, sampler, n_iterations)

