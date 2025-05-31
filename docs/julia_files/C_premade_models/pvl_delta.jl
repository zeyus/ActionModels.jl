# # The PVL-Delta

# ##  Introduction
# The PVL-Delta model is a classic model in mathematical psychology, which can be used to model reward learning over a discrete set of choices.
# It is in particular used with the Iowa Gambling Task (IGT), which is a standard task for cognitive modelling, where participants choose between four decks of cards, each with different reward and loss probabilities, and must learn which decks are better to choose from.
# The PVL-Delta model combines principles from Prospect Value Learning (PVL) models and Expectancy Valence (EV) models.
# It consists of three steps. First, the observed reward is transformed according to Prospect Theory, to make it into a subjective value $u_t$.
# This is done by transforming the value of the reward according to a power function, which is defined by the reward sensitivity parameter $A$, and adding a loss aversion parameter $w$ that controls how much losses and rewards differ in subjective value.

# $$u_t = \begin{cases} r_t^{A} & \text{if } r_t \geq 0 \\ -w \cdot |r_t|^{A} & \text{if } r_t < 0 \end{cases}$$

# where $r_t$ is the observed reward, $A$ is the reward sensitivity, and $w$ is the loss aversion.
# The next step uses a classic Rescorla-Wagner learning rule to update the expected value of the chosen option:

# $$V_{t,c} = V_{t-1,c} + \alpha \cdot (u_t - V_{t-1,c})$$

# where $V_{t,c}$ is the expected value at time $t$ for option $c$, $V_{t-1, c}$ is the expected value at time $t-1$, $\alpha$ is the learning rate, and $u_t$ is the subjective value of the reward.
# Options that were not chosen are not updated, so the expected values of all other options remain the same.
# Finally, the action probabilities are calculated using a softmax function over the expected values, weighted by a noise parameter $\beta$:

# $$P(a_t = i) = \sigma(E_{t,i} \cdot \beta)$$

# where $P(a_t = i)$ is the probability of choosing action $i$ at time $t$, $E_{t,i}$ is the expected value of action $i$ at time $t$, and $\beta$ is the action precision.
# $\sigma$ is the softmax function, which ensures that the action probabilities sum to 1, defined as:

# $$\sigma(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

# And the resulting actions denote which deck was chosen at each timestep.

# In total, the PVL-Delta model then has five parameters:
# - the learning rate $\alpha \in [0,1]$, which controls how quickly the expected values are updated
# - the reward sensitivity $A \in [0,1]$, which controls how quickly increases in the subjective value drops in relation to increases in observed reward, with $A = 1$ meaning that the subjective value is equal to the observed reward
# - the loss aversion $w \in [0, \infty]$, which controls how much losses are weighted more than gains, with $w = 1$ meaning that losses and gains are weighted equally
# - the action precision $\beta \in [0, \infty]$, which controls how much noise there is in the action selection process
# - the initial expected value $V_{0,c} \in \mathbb{R}$ for each option $c$, by default set to 0 for all actions.

# And there is one state:
# - the expected value $V_{t,c} \in \mathbb{R}$ for each option $c$ at time $t$.

# It takes two observations:
# - the index of the chosen option $c_t \in \{1, 2, \ldots, n\}$, which in the IGT is the index of the deck chosen at time $t$
# - the observed reward $r_t \in \mathbb{R}$, which is the reward received for the chosen option at time $t$.

# And returns one action:
# - the chosen option $a_t \in \{1, 2, \ldots, n\}$, which is the index of the option chosen at time $t+1$.

# Notably, in this example, as is common in many datasets, the action returned by the PVL-delta is the index of the deck chosen at that same timestep.
# This means that actions must be sampled before the reward is observed, which is ensured by setting `act_before_update = true` when creating the model. See below.
# Had it been a dataset where the action where the reward from the last trial was received in the same timestep as the action for this timestep was chosen, we would have set `act_before_update = false` when creating the model.

# ## Implementation in ActionModels
# In this section, we will demonstrate how to use the premade PVL-Delta model in ActionModels.

# We first load the ActionModels package, StatsPlots for plotting results, and DataFrames and CSV for loading data
using ActionModels
using StatsPlots
using DataFrames, CSV

# Then we create the PVL-Delta action model.
action_model = ActionModel(
    PVLDelta(
        #The number of options, which is the number of decks in the IGT
        n_options = 4,
        #The various parameters of the PVL-Delta
        learning_rate = 0.1,
        action_noise = 1,
        reward_sensitivity = 0.5,
        loss_aversion = 1,
        #The initial expected value for each option. This should be the same length as `n_options`.
        initial_value = zeros(4),
        #Set to true if the action is made before the reward is observed
        act_before_update = true, 
    ),
)

# And we can now use this action model to simulate behaviour or fit to data.
# The PVL-Delta model takes two observations: the deck chosen and the reward received.

#Create observations
observations = [
    (1, 75.0),
    (1, -50.0),
    (2, 100.0),
    (2, -25.0),
    (3, 50.0),
    (3, -75.0),
    (4, 0.0),
    (4, 25.0),
    (1, 100.0),
    (1, -100.0),
    (2, 50.0),
    (2, -50.0),
    (3, 75.0),
    (3, -25.0),
    (4, 0.0),
    (4, 100.0),
]

#Instantiate agent
agent = init_agent(action_model, save_history = true)

#Simulate behaviour
simulated_actions = simulate!(agent, observations);

#Extract the history of expected values
expected_values = get_history(agent, :expected_value)

#Collect into a Matrix
expected_values = transpose(hcat(expected_values...))

#Plot expectations over time
expectation_plot = plot(
    0:length(observations), expected_values,
    ylabel = "Expected Value",
    legend = :topright,
    label = ["Deck 1" "Deck 2" "Deck 3" "Deck 4"],
)
#Plot rewards colored by deck choice
deck_choices = [observation[1] for observation in observations]
rewards = [observation[2] for observation in observations]
rewards_plot = scatter(
    1:length(rewards), rewards;
    group = deck_choices,
    xlabel = "Timestep",
    ylabel = "Reward",
    label = nothing,
    markersize = 6,
    ylims = (minimum(rewards)-20, maximum(rewards)+20),
)
#Plot together
plot_title = plot(
    title = "Expected rewards over time by deck",
    grid = false,
    showaxis = false,
    bottom_margin = -180Plots.px,
)
plot(plot_title, expectation_plot, rewards_plot, layout = (3, 1))

#TODO: Do this with a proper IGT environment

# And finally, we can also fit it to data.
# We will do this by fitting to behavioural data from an experiment where healthy controls and subbjects with addictions to heroin and amphetamine play the IGT, by Ahn et al. (2014).
# We will here only use a subset of the data. See the [tutorial on the IGT](./example_igt.md) for more details the data, and an example of fitting the full dataset.

# First, we load the data:
ActionModels_path = dirname(dirname(pathof(ActionModels))) #hide
docs_path = joinpath(ActionModels_path, "docs") #hide
#Read in data on healthy controls
data = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_healthy_control.txt"),
    DataFrame,
)
data[!, :clinical_group] .= "control"

#Make column with total reward
data[!, :reward] = Float64.(data[!, :gain] + data[!, :loss])

#Make column with subject ID as string
data[!, :subjID] = string.(data[!, :subjID])

#Select the three first subjects
data = filter(row -> row[:subjID] in ["103", "104", "337"], data)

show(data)
# We can then create the PVL-Delta action model.
# In this case, the data is strutured so that actions are made on each timestep before the reward is observed, so we set `act_before_update = true`.
action_model = ActionModel(PVLDelta(n_options = 4, act_before_update = true))

# We can then create the full model, and sample from the posterior.
population_model = (
    learning_rate = LogitNormal(),
    action_noise = LogNormal(),
    reward_sensitivity = LogitNormal(),
    loss_aversion = LogNormal(),
)

model = create_model(
    action_model,
    population_model,
    data;
    action_cols = :deck,
    observation_cols = (chosen_option = :deck, reward = :reward),
    session_cols = :subjID,
)

chns = sample_posterior!(model)

# TODO: plot results
