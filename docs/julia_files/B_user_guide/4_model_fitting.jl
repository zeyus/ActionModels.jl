# # Fitting action models to data

# In this section, we will cover how to fit actions models to data with ActionModels.jl.
# First, we will demonstrate how to set up the full model that will be fit using the Turing.jl framework.
# This consists of specifying an action model, a population model, and the data to fit the model to.
# Then, we will show how to sample from the posterior and prior distribution of the model parameters.
# And finally, we will show the tools in ActionModels to inspect and extract the results of the model fitting.

# ## Setting up the model
# First we import ActionModels
using ActionModels

# We will here use the premade Rescorla-Wagner action model provided by ActionModels.jl. This is identical to the model described in the [defining action models](./2_defining_action_models.md) section.
action_model = ActionModel(PremadeRescorlaWagner())

# We will then specfiy the data that we want to fit the model to.
# For this example, we will use a simple manually created dataset, where three participants have completed an experiment where they must predict the next location of a moving target.
# Each participant has completed the experiment twice, in a control condition and under and experimental treatment.
using DataFrames

data = DataFrame(
    observations = repeat([1.0, 1, 1, 2, 2, 2], 6),
    actions = vcat(
        [0, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0, 0.5, 0.8, 1, 1.5, 1.8],
        [0, 2, 0.5, 4, 5, 3],
        [0, 0.1, 0.15, 0.2, 0.25, 0.3],
        [0, 0.2, 0.4, 0.7, 1.0, 1.1],
        [0, 2, 0.5, 4, 5, 3],
    ),
    id = vcat(
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
    ),
    treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
)

show(data)

# Finally, we will specify a population model, which is the model of how parameters vary between the different sessions in the data.
# There are various options when doing this, which are described in the [population models](./5_population_models.md) section.
# Here, we will use a regression population model, where we assume that the learning rate and action noise parameters depend linearly on the experimental treatment.
# It is a hierarchical model, which means assuming that the parameters are sampled from a Gaussian distribution, where the mean of the distribution is a linear function of the treatment condition.
# This is specified with standard LMER syntax, and we use a logistic link function for the learning rate to ensure that it is between 0 and 1, and an exponential link function for the action noise to ensure that it is positive.
population_model = [
        Regression(@formula(learning_rate ~ treatment + (1 | id)), logistic),  
        Regression(@formula(action_noise ~ treatment + (1 | id)), exp),        
    ]

# Finally, we can combine the three components into a full model that can be fit to the data.
# This is done with the `create_model` function, which takes the action model, population model, and data as arguments.
# Additionally, we specify which columns in the data contain the actions, observations, and session identifiers.
# This creates an ActionModels.ModelFit object, which containts the full model, and will contain the sampling results after fitting.
model = create_model(
    action_model,
    population_model,
    data;
    action_cols = :actions,
    observation_cols = :observations,
    session_cols = [:id, :treatment],
)

# If there are multiple actions or observations, we can specify them as a NamedTuple mapping each action or observation to a column in the data.
# For example, if we had two actions and two observations, we could specify them as follows:

(action_name1 = :action_column_name1, action_name2 = :action_column_name2)

# Finally, there may be missing data in the dataset.
# If actions are missing, they can be imputed by ActionModels. This is done by setting the `impute_actions` argument to `true` in the `create_model` function.
# If `impute_actions` is not set to `true`, missing actions will simply be skipped during sampling instead.
# This is a problem for action models which depend on their previous actions.

# ## Sampling from the posterior


# ## Investigating the results





# create_model
# 			- sample_posterior!
# 					- multithreading
# 					- save/resume
# 	        - sample_prior!
# 			- results API
# 				- get_sess_parameters!
# 				- get_state_trajectories!
# 			    - plots