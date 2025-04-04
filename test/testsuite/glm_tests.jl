using Test

using ActionModels
using ActionModels: MixedModels, Turing, LogExpFunctions
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake
using DataFrames, Turing


@testset "linear regression tests" begin

    #Generate dataset
    data = DataFrame(
        input = repeat([1, 1, 1, 2, 2, 2], 6),
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

    #Create agent
    agent = premade_agent("continuous_rescorla_wagner_gaussian")

    #Set samplings settings
    # sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    sampler = NUTS(-1, 0.65; adtype = AutoForwardDiff())
    n_iterations = 1000
    sampling_kwargs = (; progress = false)

    @testset "intercept only" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ 1),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)

    end
    @testset "intercept + random effect only" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ 1 + (1 | id)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "THIS IS WRONG: MISSIGN IMPLICIT INTERCEPT fixed effect only" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect and random intercept by id" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 | id)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect and random intercept by id and treatment" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 | id) + (1 | treatment)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect, random intercept + slope by treatment" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 + age | treatment)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "fixed effect, random intercept + slope by treatment" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 + age | treatment) + (1 | id)),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "THIS ERRORS: order of random effects reversed" begin
        model = create_model(
            agent,
            @formula(learning_rate ~ age + (1 | id) + (1 + age | treatment) ),
            data;
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "multiple formulas" begin
        model = create_model(
            agent,
            [
                @formula(learning_rate ~ age + (1 | id)),
                @formula(action_noise ~ age + (1 | id)),
            ],
            data;
            inv_links = [identity, LogExpFunctions.exp],
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    end

    @testset "manual prior specification" begin
        model = create_model(
            agent,
            [
                @formula(learning_rate ~ age + (1 + age | treatment) + (1 | id)),
                @formula(action_noise ~ age + (1 | id)),
            ],
            data;
            inv_links = [logistic, LogExpFunctions.exp],
            priors = [
                RegressionPrior(
                    β = [Normal(0, 1), Normal(0, 1)],
                    σ = [[LogNormal(0, 1), LogNormal(0, 1)], [LogNormal(0, 1)]],
                ),
                RegressionPrior(
                    β = Normal(0, 1),
                ),
            ],
            action_cols = [:actions],
            input_cols = [:input],
            grouping_cols = [:id, :treatment],
        )
        
        samples = sample(model, sampler, n_iterations; sampling_kwargs...)

        # agent_parameters = get_session_parameters(model, samples)
        # agent_parameter_df = summarize(agent_parameters)

        # trajectories = get_state_trajectories(model, samples, ["value", "action"])
        # state_trajectories_df = summarize(trajectories)

        # using StatsPlots
        # plot_trajectories(trajectories)

        # prior_samples = sample(model, Prior(), n_iterations; sampling_kwargs...)

        # prior_agent_parameters = get_session_parameters(model, prior_samples)

        # prior_trajectories = get_state_trajectories(model, prior_samples, ["value", "action"])
        # plot_trajectories(prior_trajectories)

        # plot_parameters(prior_samples, samples)
    end
end
