# # The Rescorla-Wagner model

# ## Introduction
# The Rescorla-Wagner model is a classical model of associative learning, which describes how expectations about outcomes are updated based on observed outcomes.
# It is widely used in reinforcement learning and cognitive psychology to model how agents learn from their environment.
# In ActionModels, the Rescorla-Wagner model is provided as a premade model. It functions as a perceptual model, and can be combined with various responsde models to make a full action model.
# In the following the Rescorla-Wagner model is described mathematically, and it's binary and categorical variants are introduced.
# In the next sections, we will demonstrate how it and its variants are implemented in ActionModels.jl.

# ### The continuous Rescorla-Wagner model
# The classic rescorla wagner model has a single changing state, the expected value $V_t$ of an outcome at time $t$.
# This is done by the Rescorla-Wagner update rule:
# $$V_t = V_{t-1} + \alpha (o_t - V_{t-1})$$
# where $o_t$ is the observed outcome at time $t$, and $\alpha$ is the learning rate.
# The expected value $V_t$ can then be used to determine the action. This is often done with a report action, where the action is sampled from a Gaussian distribution with the expected value $V_t$ as mean and a noise parameter $\beta$ as standard deviation:
# $$a_t \sim \mathcal{N}(V_t, \beta)$$
# The continuous Rescorla-Wagner model model with a Gaussian report actions therefore has a total of three parameters:
# - The initial expected value $V_0 \in \mathbb{R}$ (by default set to 0)
# - The learning rate $\alpha \in [0,1]$ (by default set to 0.1)
# - The action noise $\beta \in [0, \infty]$ (by default set to 1)

# ### Binary variant
# In its binary variant, the Rescorla-Wagner model receives a binary observation (as opposed to a continuous one), and its task is to learn the probability of the binary outcome.
# Here, the expected value $V_t$ is transformed with a logistic or sigmoid function to ensure it is between 0 and 1, before it is used to calculate the prediction error:
# $$V_t = V_{t-1} + \alpha (o_t - \sigma(V_{t-1}))$$
# where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic function.
# A binary report action is then often sampled from a Bernoulli distribution with the sigmoid-transformed expected value $V_t$ as probability. An action precision (the inverse action noise $\beta$) is used to control the noise of the action:
# $$a_t \sim \text{Bernoulli}(\sigma(V_t * \beta^{-1}))$$
# The binary Rescorla-Wagner model with a Bernoulli report action therefore has a total of three parameters:
# - The initial expected value $V_0 \in \mathbb{R}$ (by default set to 0)
# - The learning rate $\alpha \in [0,1]$ (by default set to 0.1)
# - The action noise $\beta \in [0, \infty]$ (by default set to 1)

# ### Categorical variant
# In its categorical variant, the Rescorla-Wagner model receives a categorical observation (as opposed to a continuous one), and its task is to learn the probability of each observation category occuring.
# Here the observation is transformed into a one-hot encoded vector, representing for each category whether it was observed or not.
# For each category, the expected value $V_t$ is updated with the binary Rescorla-Wagner update rule:
# $$V_{t, c} = V_{t-1, c} + \alpha (o_{t, c} - \sigma(V_{t-1, c}))$$
# where $o_{t, c}$ is the observation for category $c$ at time $t$, and $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic function.
# A categorical report action is then often sampled from a Categorical distribution with the sigmoid-transformed expected values $V_t$ as probabilities (and a noise parameter $\beta$):
# $$a_t \sim \text{Categorical}(s(V_t * \beta^{-1}))$$ 
# where s(x) is the softmax function, which ensures that the probabilities sum to 1.
# The categorical Rescorla-Wagner model with a categorical report action therefore has a total of three parameters:
# - The initial expected values $V_0 \in \mathbb{R}$, which is a vector of values for each category (by default set to a vector of zeros)
# - The learning rate $\alpha \in [0,1]$ (by default set to 0.1)
# - The action noise $\beta \in [0, \infty]$ (by default set to 1)


# ## The RescorlaWagner submodel
# The three varianta of the Rescorla-Wagner model described above are implemented as submodels in ActionModels.jl.
# Here we will demonstrate how to use each of these variants as submodels in ActionModels.jl.
# We will in each case combine them with the standard report action described above, but they can also be combined with other response models too.
# Note that the next section will show how to create these models using the convenient single-line constructor also provided by ActionModels.
# This section is therefore primarily intended to demonstrate how action models with Rescorla-Wagner submodels can be created from scratch, whcih can be necessary in more complex scenarios.

# First we load the ActionModels package, and StatsPlots for plotting the results:
using ActionModels
using StatsPlots

# ### Continuous variant
# We can then create a Rescorla-Wagner submodel with the Rescorla-Wagner constructor.
submodel = ActionModels.ContinuousRescorlaWagner(
    initial_value = 0.0,  # Initial expected value
    learning_rate = 0.1,  # Learning rate
)

# And an accompanying action model function with a Gaussian report action
model_function = function rescorla_wagner_gaussian_report(
    attributes::ModelAttributes,
    observation::Float64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Extract Rescorla-Wagner submodel
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the observation
    ActionModels.update!(rescorla_wagner, observation)

    #Extract the expected value from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Return the action distribution, which is a Gaussian with the expected value Vₜ as mean and the action noise β as standard deviation
    action_distribution = Normal(Vₜ, β)

    return action_distribution
end

# We can then create an ActionModel object
action_model = ActionModel(
    model_function,
    submodel = submodel,
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation()),
    actions = (; report = Action(Normal)),
)

# And proceed to simulate or fit the model as usual.
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = collect(0:0.1:2) .+ randn(21) * 0.1

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Plot trajectory of expected values
plot(agent, :expected_value)

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (learning_rate = LogitNormal(), action_noise = LogNormal(), initial_value = Normal()),
    observations,
    simulated_actions,
)

chns = sample_posterior!(model)

#TODO: plot the posteriors

# ### Binary variant
# We can also create a binary Rescorla-Wagner submodel:
submodel = ActionModels.BinaryRescorlaWagner(
    initial_value = 0.0,  # Initial expected value
    learning_rate = 0.1,  # Learning rate
)

# And an accompanying action model function with a Bernoulli report action.
# Now, the observation is a binary value (0 or 1), and the expected value is transformed with a sigmoid function before being used to calculate the action distribution.
model_function = function rescorla_wagner_bernoulli_report(
    attributes::ModelAttributes,
    observation::Int64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Transform the action noise into a precision (or inverse noise)
    β = 1 / β

    #Extract Rescorla-Wagner submodel
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the binary observation
    ActionModels.update!(rescorla_wagner, observation)

    #Extract the expected value from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Transform the expected value with a logistic function to get the action probability, weighted by the action precision β
    action_probability = logistic(Vₜ * β)

    #Return the action distribution, which is a Bernoulli distribution with the action probability as parameter
    action_distribution = Bernoulli(action_probability)

    return action_distribution
end

# We can then create an ActionModel object, with the observation as an integer (0 or 1) and the action as a Bernoulli distribution.
action_model = ActionModel(
    model_function,
    submodel = submodel,
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation(Int64)),
    actions = (; report = Action(Bernoulli)),
)

# And proceed to simulate or fit the model as usual.
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Plot trajectory of expected values
plot(agent, :expected_value)

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (learning_rate = LogitNormal(), action_noise = LogNormal(), initial_value = Normal()),
    observations,
    Int64.(simulated_actions),
)

chns = sample_posterior!(model)

#TODO: plot the posteriors

# ### Categorical variant
# We can also create a categorical Rescorla-Wagner submodel:
submodel = ActionModels.CategoricalRescorlaWagner(
    n_categories = 4,           # Number of categories
    initial_value = zeros(4),   # Initial expected values for each category
    learning_rate = 0.1,        # Learning rate
)

# And an accompanying action model function with a categorical report action.
# Now, the observation is a categorical value (1, 2, 3 or 4), and the expected value is transformed with a softmax function before being used to calculate the action distribution.
# We additionally rely on LogExpFunctions for the softmax function.
using LogExpFunctions

model_function = function rescorla_wagner_categorical_report(
    attributes::ModelAttributes,
    observation::Int64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Transform the action noise into a precision (or inverse noise)
    β = 1 / β

    #Extract Rescorla-Wagner submodel
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the categorical observation
    ActionModels.update!(rescorla_wagner, observation)

    #Extract the vector of expected values from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Transform the expected value with a logistic function to get the action probability, weighted by the action precision β
    action_probabilities = softmax(Vₜ .* β)

    #Return the action distribution, which is a Bernoulli distribution with the action probability as parameter
    action_distribution = Categorical(action_probabilities)

    return action_distribution
end

# We can then create an ActionModel object, with the observation as an integer (0 or 1) and the action as a Bernoulli distribution.
action_model = ActionModel(
    model_function,
    submodel = submodel,
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation(Int64)),
    actions = (; report = Action(Categorical)),
)

# And proceed to simulate or fit the model as usual.
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = [2, 1, 2, 2, 1, 2, 3, 4, 3, 2, 2, 2, 1, 1]

#Simulate actions
simulated_actions = simulate!(agent, observations)

#TODO: make plot_trajectory function compatible with multivariate states
expected_values = get_history(agent, :expected_value)
plot([expected_values[i][1] for i = 1:length(expected_values)], label = "category 1")
plot!([expected_values[i][2] for i = 1:length(expected_values)], label = "category 2")
plot!([expected_values[i][3] for i = 1:length(expected_values)], label = "category 3")
plot!([expected_values[i][4] for i = 1:length(expected_values)], label = "category 4")
title!("Expected values for each category over time")
xlabel!("Time")

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = MvNormal(zeros(4), I),
    ),
    observations,
    Int64.(simulated_actions),
)

chns = sample_posterior!(model)

#TODO: plot the posteriors



# ## The premade model constructor
# In the previous section, we have demonstrated how to use the Rescorla-Wagner submodels to create action models.
# However, ActionModels also provides a premade model constructor for the Rescorla-Wagner model, which allows you to create a full action model with a single function call.
# In this section, we will demonstrate how to use the premade model constructor to create Rescorla-Wagner action models.

# First we load ActionModels, as well as StatsPlots for plotting the results:
using ActionModels
using StatsPlots

# We can then create the configuration struct for the Rescorla-Wagner model.
rescorla_wagner_config = RescorlaWagner()

# And use that to create an action model.
action_model = ActionModel(rescorla_wagner_config)

# Which can now be used to simulate or fit the model as usual.
# By default, the RescorlaWagner constructor creates a continuous Rescorla-Wagner model with a Gaussian report action.
# It can also create binary and categorical Rescorla-Wagner models, by specifying the `type` keyword argument.
# Finally, it is possible to specify whether the expected value should be updated before or after the action is sampled, by setting the `act_before_update` keyword argument.
# This is useful with datasets that are structured so that the observation for a given timestep is not available until after the action has been sampled, such as in some reinforcement learning tasks.

continuous_RW_config = RescorlaWagner(
    type = :continuous,
    initial_value = 0.0,  # Initial expected value
    learning_rate = 0.1,  # Learning rate
    act_before_update = false,  # Whether to act before updating the expected value
)
continuous_RW = ActionModel(continuous_RW_config)

binary_RW_config = RescorlaWagner(
    type = :binary,
    initial_value = 0.0,  # Initial expected value
    learning_rate = 0.1,  # Learning rate
    act_before_update = false,  # Whether to act before updating the expected value
)
binary_RW = ActionModel(binary_RW_config)

categorical_RW_config = RescorlaWagner(
    type = :categorical,
    n_categories = 4,           # Number of categories
    initial_value = zeros(4),   # Initial expected values for each category - also sets the number of categories
    learning_rate = 0.1,        # Learning rate
    act_before_update = false,  # Whether to act before updating the expected value
)
categorical_RW = ActionModel(categorical_RW_config)

# This means that users do not have to create action model manually, but canproceed directly to simulating or fitting the model.
# With a categorical Rescorla-Wagner model, it looks like this:

#Load packages
using ActionModels
using StatsPlots

#Create action model
action_model = ActionModel(RescorlaWagner(type = :categorical, n_categories = 4))

#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = [2, 1, 2, 2, 1, 2, 3, 4, 3, 2, 2, 2, 1, 1]

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = MvNormal(zeros(4), I),
    ),
    observations,
    Int64.(simulated_actions),
)

chns = sample_posterior!(model)

# ### Custom response models
# In the examples above, we have used the default report action with the rescorla-wagner model.
# The precreated model, however, also allows for using custom response models.
# Here we show how to create the Gaussian report action as a custom response model - this is the default response model when using the continuous Rescorla-Wagner model.
# We will make a minor change, and add a bias parameter to the response model, which will be added to the expected value before sampling the action.

# The response model itself should be a function that takes the model attributes as input and returns a distribution.
# The Rescorla-Wagner will have either been updated or not, depending on the `act_before_update` keyword, so the response model should be able to handle either case.
response_model = function gaussian_report(attributes::ModelAttributes)
    rescorla_wagner = attributes.submodel
    Vₜ = rescorla_wagner.expected_value
    β = load_parameters(attributes).action_noise
    b = load_parameters(attributes).bias
    return Normal(Vₜ + b, β)
end

# In addition to the response model function, we need to specify the parameters, observations, and actions that the response model uses.
# This uses the same syntax as usual whne creating action models.
response_model_parameters = (; action_noise = Parameter(1.0), bias = Parameter(0.0))
response_model_observations = (; observation = Observation(Float64))
response_model_actions = (; report = Action(Normal))

# We can now create the Rescorla-Wagner action model with the custom response model.
action_model = ActionModel(
    RescorlaWagner(
        type = :continuous,
        response_model = response_model,
        response_model_parameters = response_model_parameters,
        response_model_observations = response_model_observations,
        response_model_actions = response_model_actions,
    ),
)

# And continue to simulate or fit the model as usual.
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

set_parameters!(agent, (; bias = 2.0))

#Create observation
observations = collect(0:0.1:2) .+ randn(21) * 0.1

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Plot trajectory of expected values
plot(agent, :expected_value)

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = Normal(),
        bias = Normal(),
    ),
    observations,
    simulated_actions,
)

chns = sample_posterior!(model)

# ### Multiple observation sources
# In some experiments, there are multiple soruces of observations. It is then a common strategy to use a single Rescorla-Wagner model to learn the expected value of each observation source, and then combine these expected values in a custom response model.
# TODO: This needs multiple submodels implemented
