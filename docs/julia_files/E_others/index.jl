# # Welcome to ActionModels! 

# ActionModels.jl is a Julia package for computational modeling of cognition and behaviour.
# It can be used to fit cognitive models to data, as well as to simulate behaviour.
# ActionModels allows for easy specification of hiearchical models, as well as including generalized linear regressions of model parameters, using standard LMER syntax.
# ActionModels makes it easy to specify new models, but also contains a growing library of precreated models, and can easily be extended to include more complex model families.
# The package is designed to have functions for every step of a classic cognitive modelling framework, such as parmeter recovery, predictive checks, and extracting trajectories of cognitive states for further analysis.
# Inspired by packages like brms and HBayesDM, ActionModels is designed to provide a flexible but intuitive and easy-to-use interface for a field that is otherwise only accessible to technical experts.
# Under the hood, ActionModels relies on Turing.jl, Julia's powerful framework for probabilistic modelling, but Julia's native automatic differentiation means that users do not have to engage directly with Turing's API.
# ActionModels is continuously being developed and optimised within the constraints of cognitive modelling. It allows for parallelizing models across experimental sessions, and can use Turing's composite samplers to estimate both continuous and discrete parameters.
# This documentation covers all three main components of ActionModels: defining cognitive models, fitting them to data, and simulating behaviour.
# It also describes how to extend or contribute to ActionModels to include new models, and how to debug models. It beings, however, with a brief theoretical introduction to the field and method of cognitive modelling.


# # Getting Started

# First we load the ActionModels package
using ActionModels

# We can now quickly define a cognitive model. We write a function that describes the action selection process in a single timestep.
# Here we create the classic Rescorla-Wagner model, with a Gaussian-noise report as action:
function rescorla_wagner(attributes::ModelAttributes, observation::Float64)
    #Read in parameters and states
    parameters = load_parameters(attributes)
    states = load_states(attributes)

    α = parameters.learning_rate
    β = parameters.action_noise
    Vₜ₋₁ = states.expected_value
    
    #The Rescorla-Wagner update rule updates the expected value U, based on the observation and the learning rate α
    Vₜ = Vₜ₋₁ + α * (observation - Vₜ₋₁)

    #The updated expected value is stored to be accessed on next timestep
    update_state!(attributes, :expected_value, Vₜ)

    #The probability distribution for the action on this timestep is a Gaussian with the expected value V as mean, and a noise parameter β as standard deviation
    action_distribution = Normal(Vₜ, β)

    return action_distribution
end;

# We now create the model object.
# We first define the attributes of the Rescorla Wagner model. This includes it's three parameters, the expected value state, the observation and the action:
# Then we use the ActionModel constructor to create the model object.
parameters = (
    learning_rate = Parameter(0.1),                             #The learning rate, with a default value of 0.1
    action_noise = Parameter(1),                                #The action noise, with a default value of 1
    initial_value = InitialStateParameter(0, :expected_value),  #And the initial expected value V₀, with a default value of 0
)
states = (;
    expected_value = State(),           #The expected value V, which is updated on each timestep
)
observations = (;
    observation = Observation()         #The observation, which is passed to the model on each timestep and used to update V
)
actions = (;
    report = Action(Normal)             #The report action, which reports the expected value with Gaussian noise
)

action_model = ActionModel(
    rescorla_wagner,
    parameters = parameters,
    states = states,
    observations = observations,
    actions = actions,
)


# We can now read in a dataset. In this example, we will use a simple simulated dataset, where three participants each have stated predictions after each of 6 observations, under some treatment condition as well as in a control condition.
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

# We can now create a model for estimating parameters hierarchically for each participant.
# We make a regression model where we estimate how much the learning rate and action noise differ between treatment conditions.
# We include a random intercept for each participant, making this a hierarchical model.
# The initial value parameter is not estimated, and is fixed to it's default: 0. 

# model = create_model(
#     action_model,
#     [@formula(learning_rate ~ treatment + (1 | id)), @formula(action_noise ~ treatment + (1 | id))],
#     data;
#     action_cols = :actions,
#     observation_cols = :observations,
#     session_cols = [:id, :treatment],
# )

model = create_model(
    action_model,
    [@formula(learning_rate ~ treatment + (1 | id))],
    data;
    action_cols = :actions,
    observation_cols = :observations,
    session_cols = [:id, :treatment],
)

# We can now fit the model to the data, extract the estimated parameters for each participant, and summarize it as a dataframe:
using StatsPlots #load statsplots for plotting results

sample_posterior!(model, progress = false);                       #Fit the model to the data
parameters_per_session = get_session_parameters!(model)           #Extract the full distribution of parameters for each participant
summarized_parameters = summarize(parameters_per_session, median) #Populate a dataframe with the median of each posterior distribution

show(summarized_parameters)

# TODO: we can plot the estimated parameters

# We can also extract the estimated value of V at each timestep, for each participant:
state_trajectories = get_state_trajectories!(model, :expected_value) #Extract the estimated trajectory of V
summarized_trajectories = summarize(state_trajectories, median)      #Summarize the trajectories

show(summarized_trajectories)

# TODO: we can also plot the estimated state trajectory

# Finally, we can also simulate behaviour using the model. 
# First we instantiate an Agent object, which produces actions according to the action model.
# Additionally, we can specify which states to save in the history of the agent.

agent = init_agent(action_model, save_history = [:expected_value]) #Create an agent object

# We can set parameter values for the agent, and simulate behaviour for some set of observations

#Set the parameters of the agent
set_parameters!(agent, (learning_rate = 0.8, action_noise = 0.01)) 

#Simulate the agent for 6 timesteps, with some the specified observations
observations = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
simulated_actions = simulate!(agent, observations) 

#Plot the change in expected value over time
plot(agent, :expected_value, label = "expected value", ylabel = "value")
plot!(observations, linetype = :scatter, label = "observation")
plot!(simulated_actions, linetype = :scatter, label = "action")