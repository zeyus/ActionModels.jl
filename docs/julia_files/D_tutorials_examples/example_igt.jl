# # Tutorial: Fitting the PVL-Delta model to data from the Iowa Gambling Task
# In this tutorial, we will fit the PVL-Delta model to data from the Iowa Gambling Task (IGT) using the ActionModels.jl package.
# In the IGT, participants choose cards from four decks, each with different reward and loss probabilities, and must learn over time which decks are advantageous.
# We will use data from Ahn et al. (2014), which includes healthy controls and participants with heroin or amphetamine addictions.
# There are more details about the collected data in the docs/example_data/ahn_et_al_2014/ReadMe.txt

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

ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344"], ahn_data)


# ## Creating the model
# Then we construct the model to be fitted to the data. 
# We use the PVL-Delta action model, which is a classic model for the IGT.
# In the PVL-Delta is a type of reinfrocement learning model that learns the expected value for each of the decks in the IGT.
# First, the observed reward is transformed with a prospect theory-based utlity curve.
# This means that the subjective value of a reward increses sub-linearly with reward magnitute, and that losses are weighted more heavily than gains.
# The expected value of each deck is then updated using a delta rule, which is the simple reinforcement learning rule used in the classic Rescorla-Wagner model.
# Finally, the probability of selecting each deck is calculated using a softmax function over the expected values of the decks, scaled by an action precision parameter.
# In summary, the PVL-Delta has four parameters: the learning rate $\alpha$, the reward sensitivity $A$, the loss aversion $w$, and the action precision $\beta$.
# See the [section on the PVL-Delta premade model] REF in the documentation for more details.

# We create the PVL-Delta using the premade model from ActionModels.jl.
# We specify the number of decks, and also that actions are selected before the expected values are updated.
# This is because in the IGT, at least as structured in this dataset, participants select a deck before they receive the reward and update expectations.
action_model = ActionModel(PVLDelta(n_options = 4, act_before_update = true))

# We then specify whcih column in the data corresponds to the action (deck choice) and which columns correspond to the observations (deck and reward).
# We also specify the columns that uniquely identify each session.
action_cols = :deck
observation_cols = (chosen_option = :deck, reward = :reward)
session_cols = :subjID

# Finally, we create the full model. We use a hierarchical regression model to predict the parameters of the PVL-Delta model based on the clinical group (healthy, heroin, or amphetamine).
# First, we will set appropriate priors for the regression coefficients.
# For the action noise and the loss aversion, the outcome of the regression will be exponentiated before it is used in the model, so pre-transformed outcomes around 2 (exp(2) ≈ 7) are among the most extreme values to be expected.
# For the learning rate and reward sensitiy, we will use a logistic transformation, so pre-transformed outcomes around around 5 (logistic(5) ≈ 0.993) are among the most extreme values to be expected.

plot(Normal(0, 0.4), label = "Intercept")
plot!(Normal(0, 0.3), label = "Effect of clinical group")
plot!(truncated(Normal(0, 0.3), lower = 0), label = "Random intercept std")
title!("Regression priors for the Rescorla-Wagner model")
xlabel!("Regression coefficient")
ylabel!("Density")

regression_prior = RegressionPrior(
    β = [Normal(0, 0.4), Normal(0, 0.3)],
    σ = truncated(Normal(0, 0.3), lower = 0),
)

model = create_model(
    action_model,
    [
        Regression(
            @formula(learning_rate ~ clinical_group + (1 | subjID)),
            logistic,
            regression_prior,
        ),
        Regression(
            @formula(reward_sensitivity ~ clinical_group + (1 | subjID)),
            logistic,
            regression_prior,
        ),
        Regression(
            @formula(loss_aversion ~ clinical_group + (1 | subjID)),
            exp,
            regression_prior,
        ),
        Regression(
            @formula(action_noise ~ clinical_group + (1 | subjID)),
            exp,
            regression_prior,
        ),
    ],
    ahn_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

# ## Fitting the model
# We are now ready to fit the model to the data.
# For this model, we will use the Enzyme automatic differentiation backend, which is a high-performance automatic differentiation library. 
# Crucially, it supports parallelization within the model, which can speed up the fitting process significantly.
# Additionally, to keep the runtime of this tutorial short, we will only fit a single chain with 500 samples.

## Set AD backend ##
using ADTypes: AutoEnzyme
import Enzyme: set_runtime_activity, Reverse
ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true));

## Fit model ##
chns = sample_posterior!(model, n_chains = 1, n_samples = 500, ad_type = ad_type)


# We can now inspect the results of the fitting process.
# Here, we just briefly plot the posterior distributions for the beta parameters.
# Since we are only using a subset of the data, we only see and indication of an effect, primarily in the learning rate.

title =
    plot(title = "Posterior over effect of clinical condition", grid = false, showaxis = false, bottom_margin = -30Plots.px)
plot(
    title,
    density(
        chns[Symbol("learning_rate.β[2]")],
        title = "learning rate",
        label = nothing,
    ),
    density(
        chns[Symbol("reward_sensitivity.β[2]")],
        title = "reward sensitivity ",
        label = nothing,
    ),
    density(
        chns[Symbol("loss_aversion.β[2]")],
        title = "loss aversion ",
        label = nothing,
    ),
    density(
        chns[Symbol("action_noise.β[2]")],
        title = "action noise",
        label = nothing,
    ),
    layout = @layout([A{0.01h}; [B C ; D E]])
)

# We can also extract the session parameters and state trajectories from the model.
session_parameters = get_session_parameters!(model)
state_trajectories = get_state_trajectories!(model, :expected_value)

# We can investigate the estimated session parameters in more detail

#TODO: plot the session parameters

#And we can extract a dataframe with the median of the posterior for each session parameter
parameters_df = summarize(session_parameters)
show(parameters_df)

# We can also look at the implied state trajectories, in this case the expected value.
# These can be plotted, or they can be summarized in a dataframe, which can then be used for further analysis, such as correlating with physiological states.

#TODO: plot the state trajectories

states_df = summarize(state_trajectories)
show(states_df)




# ## Comparing to a simple random model
# We can also compare the PVL-Delta to a simple random model, which simply samples actions from a fixed Categorical distribution.

# ### Creating the Gaussian random action model 
# In this model, actions are sampled from a Gaussian distribution with a fixed mean and standard deviation.
# This meanst that there are two parameters in the action model: the action noise $\beta$ and the action mean $\mu$.

function gaussian_random(agent::Agent, input::T) where {T<:Real}

    β = agent.parameters[:action_noise]
    μ = agent.parameters[:mean]

    return Normal(μ, β)
end

action_model = ActionModel(
    gaussian_random,
    parameters = (action_noise = Parameter(1), mean = Parameter(50)),
)

# ### Fitting the model
# We also set priors hfor this simpler model.
# Here we set the priors separately for the mean and the noise, since they are on very different scales.
# We center the priors for the mean at 50, as that is the iddle of the range of actions.
# The priors for the noise are similar to those used with the Rescorla-Wagner model.

mean_regression_prior = RegressionPrior(
    β = [Normal(50, 10), Normal(0, 10)],
    σ = truncated(Normal(0, 10), lower = 0),
)
noise_regression_prior = RegressionPrior(
    β = [Normal(0, 0.3), Normal(0, 0.5)],
    σ = truncated(Normal(0, 0.3), lower = 0),
)

population_model = [
    Regression(@formula(mean ~ 1 + pdi_total + (1 | ID)), mean_regression_prior),
    Regression(
        @formula(action_noise ~ 1 + pdi_total + (1 | ID)),
        exp,
        noise_regression_prior,
    ),
]

simple_model = create_model(
    action_model,
    population_model,
    JGET_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

## Set AD backend ##
using ADTypes: AutoEnzyme
import Enzyme: set_runtime_activity, Reverse
ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true));

## Fit model ##
chns = sample_posterior!(simple_model, n_chains = 1, n_samples = 500, ad_type = ad_type)

#TODO: plot the results

#TODO: model comparison




# ### CATEGORICAL RANDOM ####
# function categorical_random(agent::Agent, input::Tuple{Int64,Float64})

#     deck, reward = input

#     action_noise = exp(agent.parameters[:action_noise])

#     #Set the probability 
#     base_probs = [0.1, 0.4, 0.1, 0.1]

#     #Do a softmax of the values
#     action_probabilities = softmax(base_probs * action_noise)

#     return Categorical(action_probabilities)
# end

# action_model = ActionModel(
#     categorical_random,
#     parameters = (; action_noise = Parameter(1)),
#     observations = (deck = Observation(Int64), reward = Observation(Float64)),
#     actions = (; deck = Action(Categorical)),
# )


# model = create_model(
#     action_model,
#     Regression(@formula(action_noise ~ clinical_group + (1 | subjID)), exp),
#     ahn_data,
#     action_cols = action_cols,
#     observation_cols = observation_cols,
#     session_cols = session_cols,
# )

# chains = sample_posterior!(model)