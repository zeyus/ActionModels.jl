# # The PVL-Delta model

# ##  Introduction
# The PVL-Delta model is a classic model in mathematical psychiatry, which can be used to model reward learning over a discrete set of choices.
# It is in particular used with the Iowa Gambling Task (IGT), which is a standard task for cognitive modelling.
# The PVL-Delta model combines principles from Prospect Value Learning (PVL) models and Expectancy Valence (EV) models.
# It consists of three steps. First, the observed reward is transformed according to Prospect Theory, to make it into a subjective value.
# This is done by transofrming the value of the reward according to a power function, which is defined by the reward sensitivity parameter $A$, and adding a loss aversion parameter $w$ that controls how much losses and rewards differ in subjective value.
# $$ u_t = \begin{cases}
#     r_t^{A} & \text{if } r_t \geq 0 \\
#     -w \cdot |r_t|^{A} & \text{if } r_t < 0
# \end{cases} $$
# where $r_t$ is the observed reward, $A$ is the reward sensitivity, and $w$ is the loss aversion.
# The next step uses a classic Rescorla-Wagner learning rule to update the expected value of each choice, which is defined as:
# $$ E_t = E_{t-1} + \alpha \cdot (u_t - E_{t-1}) $$
# where $E_t$ is the expected value at time $t$, $E_{t-1}$ is the expected value at time $t-1$, $\alpha$ is the learning rate, and $u_t$ is the subjective value fo the reward.
# Finally, the action probabilities are calculated using a softmax function over the expected values, weighted by a noise parameter $\beta$ which is defined as:
# $$P(a_t = i) = \sigma(E_{t,i} \cdot \beta)$$
# where $P(a_t = i)$ is the probability of choosing action $i$ at time $t$, $E_{t,i}$ is the expected value of action $i$ at time $t$, and $\beta$ is the action precision.
# $\sigma$ is the softmax function, which ensures that the action probabilities sum to 1, defined as:
# $$ \sigma(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$
# In total, the PVL-Delta model then has five parameters:
# - the learning rate $\alpha \in [0,1]$, which controls how quickly the expected values are updated
# - the reward sensitivity $A \in [0,1]$, which controls how quickly increases in the subjective value drops in relation to increases in observed reward, with $A = 1$ meaning that the subjective value is equal to the observed reward
# - the loss aversion $w \in [0, \infty)$, which controls how much losses are weighted more than gains, with $w = 1$ meaning that losses and gains are weighted equally
# - the action precision $\beta \in [0, \infty)$, which controls how much noise there is in the action selection process
# - the initial expected value $Ev_0 \in \mathbb{R}$ for each possible action, by default set to 0 for all actions.
# And there is one state:
# - the expected value $Ev_t \in \mathbb{R}$ for each possible action at time $t$.

# ## Implementation in ActionModels
# In this section, we will demonstrate how to use the premade PVL-Delta model in ActionModels.jl.

# We first load the ActionModels package, StatsPlots for plotting results, and DataFrames and CSV for loading data
using ActionModels
using StatsPlots
using DataFrames, CSV

# Then we create the PVL-Delta action model.
action_model = ActionModel(
    PVLDelta(
        n_options = 4,
        initial_value = zeros(4),
        learning_rate = 0.1,
        action_noise = 1,
        reward_sensitivity = 0.5,
        loss_aversion = 1,
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
agent = init_agent(action_model)

simulated_actions = simulate!(agent, observations)

#TODO: Do this with a proper environment

# And finally, we can also fit it to data.
# We will do this by fitting to behavioural data from an experiment where healthy controls and subbjects with addictions to heroin and amphetamine play the IGT, by Ahn et al. (2014).
# We will here only use a subset of the data. See the [tutorial on the IGT](./example_igt.md) for more details the data, and an example of fitting the full dataset.

# First, we load the data.
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
