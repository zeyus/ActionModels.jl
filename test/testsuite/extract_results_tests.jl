using Test
using ActionModels, DataFrames

@testset "fitting-API" begin

    ### SETUP ###
    #Generate dataset
    data = DataFrame(
        inputs = repeat([1, 1, 1, 2, 2, 2], 6),
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

    #Define input and action cols
    input_cols = [:inputs]
    action_cols = [:actions]
    grouping_cols = [:id, :treatment]

    #Create agent
    agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

    prior = Dict(
        "learning_rate" => LogitNormal(),
        "action_noise" => LogNormal(),
        "initial_value" => Normal(),
    )

    #Create model
    model = create_model(
        agent,
        prior,
        data,
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
    )


    ### TESTS ###
    posterior_chains = sample_posterior!(model)
    posterior_parameters = get_session_parameters!(model, :posterior)
    summarize(posterior_parameters)
    summarize(posterior_parameters, mean)
    posterior_trajectories = get_state_trajectories!(model, ["input", "value"], :posterior)
    summarize(posterior_trajectories)
    summarize(posterior_trajectories, mean)

    prior_chains = sample_prior!(model)
    prior_parameters = get_session_parameters!(model, :prior)
    summarize(prior_parameters)
    prior_trajectories = get_state_trajectories!(model, ["input", "value"], :prior)
    summarize(prior_trajectories)

end


