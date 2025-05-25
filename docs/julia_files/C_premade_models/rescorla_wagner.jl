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
# $V_t = V_{t-1} + \alpha (o_t - V_{t-1})$
# where $o_t$ is the observed outcome at time $t$, and $\alpha$ is the learning rate.
# The expected value $V_t$ can then be used to determine the action. This is often done with a report action, where the action is sampled from a Gaussian distribution with the expected value $V_t$ as mean and a noise parameter $\beta$ as standard deviation:
# $a_t \sim \mathcal{N}(V_t, \beta)$

# ### Binary variant
# In its binary variant, the Rescorla-Wagner model receives a binary observation (as opposed to a continuous one), and its task is to learn the probability of the binary outcome.
# Here, the expected value $V_t$ is transformed with a logistic or sigmoid function to ensure it is between 0 and 1, before it is used to calculate the prediction error:
# $V_t = V_{t-1} + \alpha (o_t - \sigma(V_{t-1}))$
# where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic function.
# A binary report action is then often sampled from a Bernoulli distribution with the sigmoid-transformed expected value $V_t$ as probability:
# $a_t \sim \text{Bernoulli}(\sigma(V_t))$

# ### Categorical variant
# In its categorical variant, the Rescorla-Wagner model receives a categorical observation (as opposed to a continuous one), and its task is to learn the probability of each observation category occuring.
# Here the observation is transformed into a one-hot encoded vector, representing for each category whether it was observed or not.
# For each category, the expected value $V_t$ is updated with the binary Rescorla-Wagner update rule:
# $V_{t, c} = V_{t-1, c} + \alpha (o_{t, c} - \sigma(V_{t-1, c}))$
# where $o_{t, c}$ is the observation for category $c$ at time $t$, and $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic function.
# TODO: check whether expected values should be transformed with a softmax function before saving them.
# A categorical report action is then often sampled from a Categorical distribution with the sigmoid-transformed expected values $V_t$ as probabilities (and a noise parameter $\beta$):
# $a_t \sim \text{Categorical}(s(V_t * \beta))$ 
# where s(x) is the softmax function, which ensures that the probabilities sum to 1.

# ## Implementation in ActionModels.jl
# Here we will demonstrate how to implement each of these variants in ActionModels.jl.
# First we load the ActionModels package, and StatsPlots for plotting the results:
using ActionModels
using StatsPlots

# ### Continuous variant
# We can then create a Rescorla-Wagner submodel with the Rescorla-Wagner constructor.
submodel = ActionModels.RescorlaWagner(
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
agent = init_agent(action_model, save_history = :expected_value)

observations = collect(0:0.1:2) .+ randn(21) * 0.1

simulate!(agent, observations)

plot(agent, :expected_value)

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
agent = init_agent(action_model, save_history = :expected_value)

observations = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

simulate!(agent, observations)

plot(agent, :expected_value)

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
agent = init_agent(action_model, save_history = :expected_value)

observations = [2, 1, 2, 2, 1, 2, 3, 4, 3, 2, 2, 2, 1, 1]

simulate!(agent, observations)

#TODO: make plot_trajectory function compatible with multivariate states

expected_values = get_history(agent, :expected_value)

plot([expected_values[i][1] for i = 1:length(expected_values)], label = "category 1")
plot!([expected_values[i][2] for i = 1:length(expected_values)], label = "category 2")
plot!([expected_values[i][3] for i = 1:length(expected_values)], label = "category 3")
plot!([expected_values[i][4] for i = 1:length(expected_values)], label = "category 4")
title!("Expected values for each category over time")
xlabel!("Time")





# ## The premade model constructor

# ### Custom response model

# ### Multiple observation sources
# In some experiments, there are multiple soruces of observations. It is then a common strategy to use a single Rescorla-Wagner model to learn the expected value of each observation source, and then combine these expected values in a custom response model.
# TODO: This needs multiple submodels implemented

