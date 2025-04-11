using Test
using ActionModels, DataFrames

@testset "inference API" begin

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

    @testset "sampling and extracting results" begin
        #Posterior
        posterior_chains = sample_posterior!(model)
        posterior_parameters = get_session_parameters!(model, :posterior)
        summarize(posterior_parameters)
        summarize(posterior_parameters, mean)
        posterior_trajectories = get_state_trajectories!(model, ["input", "value"], :posterior)
        summarize(posterior_trajectories)
        summarize(posterior_trajectories, mean)

        #Prior
        prior_chains = sample_prior!(model)
        prior_parameters = get_session_parameters!(model, :prior)
        summarize(prior_parameters)
        prior_trajectories = get_state_trajectories!(model, ["input", "value"], :prior)
        summarize(prior_trajectories)
    end

    @testset "plotting results" begin
        ### Test plotting functions ###
        using StatsPlots

        #Default Turing plots
        plot(model.posterior.chains)
        plot(model.prior.chains)

        # plot(posterior_parameters)
        # plot(posterior_parameters, session = "id:Hans.treatment:control")
        # plot(posterior_trajectories)
        # plot(posterior_trajectories, session = "id:Hans.treatment:control")

        # plot(prior_parameters)
        # plot(prior_parameters, session = "id:Hans.treatment:control")
        # plot(prior_trajectories)
        # plot(prior_trajectories, session = "id:Hans.treatment:control")
    end
    
    @testset "sample_posterior! variations" begin

        @testset "different init_params" begin
            posterior_chains = sample_posterior!(model, resample = true, init_params = nothing)
            posterior_chains = sample_posterior!(model, resample = true, init_params = :MLE)
            posterior_chains = sample_posterior!(model, resample = true, init_params = :MAP)
            posterior_chains = sample_posterior!(model, resample = true, init_params = :sample_prior)
        end

        @testset "different AD backends" begin
            for AD in [AutoForwardDiff(), AutoReverseDiff(), AutoReverseDiff(; compile=true), AutoMooncake(; config = nothing)]
                @testset "AD: $AD" begin
                    posterior_chains = sample_posterior!(model, resample = true, adtype = AD)
                end
            end
        end

        @testset "save/resume THIS ERRORS!" begin
            # posterior_chains = sample_posterior!(model, resample = true, save_resume = SampleSaveResume(path = mktempdir()))
        end

        @testset "parallel sampling" begin
            using Distributed

            addprocs(2)

            @everywhere using ActionModels
            @everywhere model = $model

            posterior_chains = sample_posterior!(model, MCMCDistributed(), resample = true)

            #TODO: test parallel sampling with save/resume

            rmprocs(workers())
        end
    end
end


