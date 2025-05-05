using ActionModels, DataFrames

## Define model ##
action_model = ActionModel(ContinuousRescorlaWagnerGaussian())

action_model = ActionModel(PremadeRescorlaWagner())

## Simulate with agent ##
agent = init_agent(action_model, save_history = [:value])

simulate!(agent, [1.,0,0,1])

using StatsPlots
plot(agent, :value)
plot(agent, :observation)

get_parameters(agent)
set_parameters!(agent, :learning_rate, 0.2)
set_parameters!(agent, :initial_value, 0.2)
get_parameters(agent)

reset!(agent)

get_states(agent)

## Fit model ##
#Generate dataset
data = DataFrame(
    observations = repeat([1., 1, 1, 2, 2, 2], 6),
    actions = vcat(
        [0, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0, 0.5, 0.8, 1, 1.5, 1.8],
        [0, 2, 0.5, 4, 5, 3],
        [0, 0.1, 0.15, 0.2, 0.25, 0.3],
        [0, 0.2, 0.4, 0.7, 1.0, 1.1],
        [0, 2, 0.5, 4, 5, 3],
    ),
    age = vcat(
        repeat([20], 6),
        repeat([24], 6),
        repeat([28], 6),
        repeat([20], 6),
        repeat([24], 6),
        repeat([28], 6),
    ),
    id = vcat(
        repeat(["Hans"], 6),
        repeat(["Georg"], 6),
        repeat(["Jørgen"], 6),
        repeat(["Hans"], 6),
        repeat(["Georg"], 6),
        repeat(["Jørgen"], 6),
    ),
    treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
)

#Define observation and action cols
observation_cols = [:observations]
action_cols = [:actions]
grouping_cols = [:id, :treatment]


#Create and fit model
prior = (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = Normal(),
    )

#Create model
model = create_model(
    action_model,
    prior,
    data,
    observation_cols = observation_cols,
    action_cols = action_cols,
    grouping_cols = grouping_cols,
)

sample_posterior!(model, n_chains = 1, n_samples = 10)
get_session_parameters!(model, :posterior)
get_state_trajectories!(model, :value, :posterior)
