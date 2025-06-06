# # Simulating states and behaviour

# In this section, we will cover how to simulate behaviour and state trajectories using an action model.
# This will consist in agent-based simulation, where we construct an agent that will act based on the action model it uses.
# First we will create an Agent object which uses the Rescorla-Wagner Gaussian noise report action model.
# Then we will show how to manipulate and extract the agent's parameters and states.
# In the next section, we will provide the agent with an environment and simulate its behaviour in that environment.

# ## Instantiating and manipulating agents
# First we import ActionModels, and create the premade Rescorla-Wagner action model.
# This model is identical to the model used in the defining action models REF section.
using ActionModels

action_model = ActionModel(RescorlaWagner())

# We can now instantiate an Agent object using this action model. 
# We can specify the states for which we want to save the history when simulating. Here we choose the `expected_value`.
# If we set `save_history` to false (the default), no history will be saved. If we set it to `true`, all states will be saved. 

agent = init_agent(action_model, save_history = :expected_value)

# We can extract parameter and state values from the action model. They are returned as NamedTuples. 
# We can see that the parameters are set to their default values:

parameters = get_parameters(agent)

# And that the states are set to their initial values:

states = get_states(agent)

# We can also get specific parameters and states by passing their names as a second argument.
#This returns the value of the learning rate parameter.
get_parameters(agent, :learning_rate)
#This returns the values of the specified parmaeters as a NamedTuple
get_parameters(agent, (:learning_rate, :action_noise))

# We can also set parameters and states. 
# Here we set the learning rate to 0.5 and the initial value of the expected value state to 0.5.
# We also set the current expected value to 0.8.
set_parameters!(agent, (learning_rate = 0.5, initial_value = 0.5, action_noise = 0.1))
set_states!(agent, (; expected_value = 0.8))

# ## Agent-based simulation
# We are now ready to simulate the agent's behaviour and expectation updates.
# First we create a vector of observations that the agent will receive.
# This is the simplest type of environment, where the observations are pre-specified and independent of the agent's actions.
# For this demonstration, we will create an increasing vector of observations, each with some noise added.
#0 to 2 in steps of 0.1, with some noise added
observations = collect(0:0.1:2) .+ randn(21) * 0.1;

# We can now simulate the agent's behaviour in this environment with the simulate! function. One action is simulated for each timestep in the observations vector.
simulated_actions = simulate!(agent, observations);

# We can also pass observations one timestep at a time with the observe! function.
single_action = observe!(agent, 0.6);

# These functions call the action model function on each timestep, and sample from the action distribution returned by the action model.
# Notably, if the action models take multiple observations, we can pass the observations as a vector of tuples, or as a matrix where each row is one observation.
# If the action model returns multiple actions, `simulate!` will return a vector of tuples as the actions.

# We can extract the trajectory of expected values from the agent's history. 
# If we don't specify a state name, the history of all states will be returned.
# Note that the history is only saved for a state if we specified it in the `save_history` argument when initializing the agent.
expected_value_trajectory = get_history(agent, :expected_value);

# The `reset!`function` empties the agent's history, and resets it's states to their initial values.
# If there are any intial state parameters, they decide the value of the states after reset.
reset!(agent)

# The initial value is now 0.5, since we set the inital state parameter `intial_value` to that value.
get_states(agent, :expected_value)

# ActionModels also provide a convenient way to plot the trajectories of the agent's states.
# We need to import the StatsPlots package for this.
using StatsPlots

#First we simulate again, since the agent was reset
simulated_actions = simulate!(agent, observations)

#The plot function take all standard plotting arguments
plot(agent, :expected_value, label = "expected value", color = :green)
#We can plot the observations and actions on the same plot
plot!(observations, label = "observation", color = :red, linetype = :scatter)
plot!(simulated_actions, label = "reported expectation", color = :blue, linetype = :scatter)
ylabel!("Value")
title!("Rescorla-Wagner expectation trajectory")

# And we can see that the Rescorla-Wagner model updates the expected value based on the observations, and reports it with Gaussian noise.
