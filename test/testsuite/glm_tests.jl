using Test

using ActionModels, DataFrames, LogExpFunctions
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake

@testset "linear regression tests" begin

    #Generate dataset
    data = DataFrame(
        observation = repeat([0.1, 1, 1, 2, 2, 2], 6),
        action = vcat(
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
    observation_cols = [:observation]
    action_cols = [:action]
    grouping_cols = [:id, :treatment]

    #Create action model
    action_model = ActionModel(ContinuousRescorlaWagnerGaussian())

    #Inference parameters
    n_samples = 200
    n_chains = 2

    #Go through each supported AD type
    for AD in
        ["AutoForwardDiff", "AutoReverseDiff", "AutoReverseDiff(true)", "AutoMooncake"]

        #Select appropriate AD backend
        if AD == "AutoForwardDiff"
            ad_type = AutoForwardDiff()
        elseif AD == "AutoReverseDiff"
            ad_type = AutoReverseDiff()
        elseif AD == "AutoReverseDiff(true)"
            ad_type = AutoReverseDiff(; compile = true)
        elseif AD == "AutoMooncake"
            ad_type = AutoMooncake(; config = nothing)
        end

        @testset "intercept only ($ad_type)" begin
            model = create_model(
                action_model,
                @formula(learning_rate ~ 1),
                data;
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

        end
        @testset "intercept + random effect only ($ad_type)" begin
            model = create_model(
                action_model,
                @formula(learning_rate ~ 1 + (1 | id)),
                data;
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
            @test sort(posterior_parameters_df, :learning_rate).id ==
                  ["Hans", "Hans", "Georg", "Georg", "Jørgen", "Jørgen"]
        end

        @testset "THIS IS WRONG: MISSING IMPLICIT INTERCEPT fixed effect only ($ad_type)" begin
            #TODO: fix this
            # model = create_model(
            #     action_model,
            #     @formula(learning_rate ~ age),
            #     data;
            #     action_cols = action_cols,
            #     observation_cols = observation_cols,
            #     grouping_cols = grouping_cols,
            # )

            # samples = sample(model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "fixed effect and random intercept by id ($ad_type)" begin
            model = create_model(
                action_model,
                @formula(learning_rate ~ age + (1 | id)),
                data;
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
        end

        @testset "fixed effect and random intercept by id and treatment ($ad_type)" begin
            model = create_model(
                action_model,
                @formula(learning_rate ~ age + (1 | id) + (1 | treatment)),
                data;
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
        end

        @testset "fixed effect, random intercept + slope by treatment ($ad_type)" begin
            model = create_model(
                action_model,
                @formula(learning_rate ~ age + (1 + age | treatment)),
                data;
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
        end

        @testset "fixed effect, random intercept + slope by treatment ($ad_type)" begin
            model = create_model(
                action_model,
                @formula(learning_rate ~ age + (1 + age | treatment) + (1 | id)),
                data;
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
        end

        @testset "THIS ERRORS: order of random effects reversed ($ad_type)" begin
            #TODO: fix this

            # model = create_model(
            #     action_model,
            #     @formula(learning_rate ~ age + (1 | id) + (1 + age | treatment)),
            #     data;
            #     action_cols = action_cols,
            #     observation_cols = observation_cols,
            #     grouping_cols = grouping_cols,
            # )

            # samples = sample(model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "multiple formulas ($ad_type)" begin
            model = create_model(
                action_model,
                [
                    @formula(learning_rate ~ age + (1 | id)),
                    @formula(action_noise ~ age + (1 | id)),
                ],
                data;
                inv_links = [identity, LogExpFunctions.exp],
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
        end

        @testset "manual prior specification ($ad_type)" begin
            model = create_model(
                action_model,
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
                    RegressionPrior(β = Normal(0, 1)),
                ],
                action_cols = action_cols,
                observation_cols = observation_cols,
                grouping_cols = grouping_cols,
            )

            #Posterior
            posterior_chains = sample_posterior!(
                model,
                ad_type = ad_type,
                n_samples = n_samples,
                n_chains = n_chains,
            )

            #Check that the posteriors are correct
            posterior_parameters = get_session_parameters!(model, :posterior)
            posterior_parameters_df = summarize(posterior_parameters)
        end
    end
end
