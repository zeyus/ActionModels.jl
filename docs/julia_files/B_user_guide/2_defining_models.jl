# # Defining models with ActionModels

# In this section, we will demonstrate how to define an action model with ActionModels.
# We will demonstrate this by first constructing a classic Rescorla-Wagner model. 
# In the next section, we will describe the possible Various options when constructing action models, which can be relevant depending on the specific type of model.
# See the section on submodels REF for using custom submodels, which can be useful when creating more complex families.
# See the seciton on premade models REF to see the growing library of prespecified models that ActionModels provides.

# ## Defining a classic Gaussian report Rescorla-Wagner model

# The Rescorla-Wagner (Wagner & Rescorla, 1972) model is a classic reinforcement learning model that describes how expectations about the environment are updated based on the observed outcome. 
# It was originally used specifically for modeling how values of taking different actions are learnt in classic conditioning tasks, but has since been used in many other learning tasks.
# In this model, the expected observation at time $t$ is denoted as $V_t$, and it is updated based on the observed outcome $o_t$ and a learning rate $\alpha$, with the following equation:

# $V_t = V_{t-1} + \alpha (o_t - V_{t-1})$

# In the Gaussian report action model, the expected value $V_t$ is used to determine the mean of a Gaussian distribution from which the action is sampled:

# $a_t \sim \mathcal{N}(V_t, \beta)$

# To implement this model in ActionModels, we will first define a function that carries out the update and returns a probability distribution for the action.
# The function should take a ModelAttributes object as its first argument, which contains the parameters, states and previous actions of the model.
# The observations to the model are passed as the following positional arguments. For this model, there is only a single continuous observation.

using ActionModels

function rescorla_wagner(attributes::ModelAttributes, observation::Float64)

    #Read in parameters and states
    parameters = load_parameters(attributes)
    states = load_states(attributes)

    α = parameters.learning_rate
    β = parameters.action_noise
    Vₜ₋₁ = states.expected_value

    #The Rescorla-Wagner update rule
    Vₜ = Vₜ₋₁ + α * (observation - Vₜ₋₁)

    #The updated expected value is stored to be accessed on next timestep
    update_state!(attributes, :expected_value, Vₜ)


    #The Gaussian report action
    action_distribution = Normal(Vₜ, β)

    return action_distribution
end;

# In the first part of the function, we read in the parameters and states of the model from the model attributes.
# In the second part, we implement the Rescorla-Wagner update rule, which updates the expected value $V_t$ based on the observation $o_t$ and the learning rate $\alpha$. Then we store the updated $V_t$ in the attributes.
# In the last part, we create a Gaussian distribution with the updated expected value $V_t$ as mean and the action noise $\beta$ as standard deviation, and return this as the probability distribution for the action for this timestep.

# Now we specify the parameters, states, observations and actions of the model.
# The first two parameters are the learning rate and action noise. The value passed to the `Parameter()` constructor is the default value for the parameter.
# We also specify an initial state parameter, which is a parameter that sets the initial value of some state. 
# In this case, the `initial_value` parameter controls the initial expectation $V_0$.
# Apart from the default parameter value, initial state parameters need to be given the state it is associated with as the second argument.

parameters = (
    #The learning rate, with a default value of 0.1
    learning_rate = Parameter(0.1),
    #The action noise, with a default value of 1                          
    action_noise = Parameter(1),
    #And the initial expected value V₀, with a default value of 0                     
    initial_value = InitialStateParameter(0, :expected_value),
);

# We then specify the states of the model. There is only one state in this model, the expected value $V_t$, which is updated on each timestep as described above.
states = (;
    #The expected value V, which is updated on each timestep
    expected_value = State(),
);

# Then we specify the observations and actions of the model.
# The observation is a single continuous observation. When specifying the action, we indicate which type of distribution the action is sampled from. 
observations = (;
    #The observation, which is passed to the model on each timestep and used to update V
    observation = Observation()
);
actions = (;
    #The report action, which reports the expected value with Gaussian noise
    report = Action(Normal)
);

# Finally, we create the model object using the `ActionModel` constructor.
action_model = ActionModel(
    rescorla_wagner,
    parameters = parameters,
    states = states,
    observations = observations,
    actions = actions,
)



# ## Other options when defining action models

# ### Multiple actions and observations
# It is possible for action models to have multiple observations or multiple actions. 
# Multiple observations just have to be specified as additional arguments in the action model function.
# Multiple actions just need to be returned as a tuple from the action model function.
# In both cases, multiple observations and actions need to be specified when constructing the `ActionModel` object.

#Dummy example function with multiple observations and actions
function example_actionmodel(
    attributes::ModelAttributes,
    first_observation::Float64,
    second_observation::Int64,
)

    #Model definition not shown here

    #Two example actions sampled from a Gaussian and a Bernoulli distribution
    first_action_distribution = Normal(0, 1)
    second_action_distribution = Bernoulli(0.5)

    return (first_action_distribution, second_action_distribution)
end

observations =
    (; first_observation = Observation(), second_observation = Observation(discrete = true));
actions = (; first_action = Action(Normal), second_action = Action(Bernoulli));

#ActionModel object creation not shown here

# ### Types of parameters, actions, observations and states
# Observations and states can be of any type, and their type can be passed to the respective constructors, or the `discrete` keyword argument can be set to `true` to indicate that the observation or state is discrete.
# Parameters and actions need to be a subtype of Real, however; i.e., either discrete or continuous. By default, they are continuous.
# Parameters can be specified to be discrete when defining them; action types are automatically inferred from the distribution type specified in the `Action()` constructor.
# Both actions and parameters can also be multivariate, i.e., they can be vectors or matrices.
# To specfiy a multivariate parameter, just specify an array as the default value in the `Parameter()` constructor.
# To specify a multivariate action, just specify a multivariate distribution in the `Action()` constructor.
# Note that states will have `missing` as their initial value if an initial value is not passed to the `State()` constructor and an InitialStateParameter is not specified.

parameters = (
    #A discrete parameter with a default value of 0
    discrete_parameter = Parameter(0, discrete = true),
    #A multivariate parameter with a default value of [0, 0]
    multivariate_parameter = Parameter([0, 0]),
);

actions = (
    #A discrete action sampled from a Bernoulli distribution
    discrete_action = Action(Bernoulli),
    #A multivariate action sampled from a multivariate normal distribution              
    multivariate_action = Action(MvNormal),
    #If different types of distributiosn can be used, it is possible to specify an abstract type of distribution.
    multivariate_action_2 = Action(Distribution{Multivariate,Continuous}),
);

# ### Depending on previous actions
# Some models can depend on previous actions.
# Similar to how parameters and states are loaded within the action model function, previous actions can be loaded using the `load_actions()` function.
# If there are multiple actions, the previous actions are returned as a named tuple.

#Dummy example function with multiple observations and actions
function example_actionmodel(attributes::ModelAttributes, observation::Float64)

    #Load previous actions
    prev_actions = load_actions(attributes)

    first_action = prev_actions.first_action
    second_action = prev_actions.second_action

    #Rest of model definition not shown here

    #The first action is sampled from a distribution that depends on the previous action
    first_action_distribution = Normal(first_action, 1)
    second_action_distribution = Bernoulli(0.5)

    return (first_action_distribution, second_action_distribution)
end;

# ### Custom conditions for parameter rejection
# Finally, it can in some cases be useful to have a custom condition where the model is not defined, so that parameter estimation can avoid parameter regions that result in this condition happening.
# Notably, this slows down inference, and is not recommended unless necessary.
# This can be done by throwing the specific error type RejectParameters within the action model function, which is then handled appropriately by the inference algorithm.

#Dummy example function with custom parameter rejection
function example_actionmodel(attributes::ModelAttributes, observation::Float64)

    if somecondition
        #If the condition is met, throw a ParameterRejectionError
        throw(RejectParameters("The model is not defined for this parameter set"))
    end

    #Rest of model definition not shown here#
end;
