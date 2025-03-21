using Test
using ActionModels, DataFrames
using Distributed

@testset "fit model" begin

    ### SETUP ###
    #Create dataframe
    data = DataFrame(
        inputs = rand([0, 1], 20),
        inputs_2 = rand([0, 1], 20),
        actions = rand([0, 1], 20),
        actions_2 = rand([0, 1], 20),
        ID = [repeat(["A"], 5); repeat(["B"], 5); repeat(["A"], 5); repeat(["B"], 5)],
        category = [repeat(["X"], 10); repeat(["Y"], 10)],
    )

    #Create agent
    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    #Set prior
    prior = Dict(
        "learning_rate" => LogitNormal(0.0, 1.0),
        "action_precision" => truncated(Normal(0.0, 1.0), lower = 0),
        ("initial", "value") => Normal(0.0, 1.0),
    )

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    n_iterations = 10
    n_chains = 2
    sampling_kwargs = (; progress = false)

    @testset "basic run" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
        )

        results = fit_model(
            model;
            sampler = sampler,
            n_iterations = n_iterations,
            n_chains = n_chains,
            sampling_kwargs...,
        )

        @test results isa ActionModels.FitModelResults
    end

    @testset "basic run - save_resume" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :ID,
        )
        save_resume = ChainSaveResume(path = mktempdir())
        results = fit_model(
            model;
            sampler = sampler,
            n_iterations = n_iterations,
            n_chains = n_chains,
            save_resume=save_resume,
            sampling_kwargs...,
        )

        @test results isa ActionModels.FitModelResults
    end


    @testset "parallelized" begin
        addprocs(4)
        @everywhere begin
            ### SETUP ###
            using ActionModels, DataFrames
            #Turing.setprogress!(false)
            #Create dataframe
            data = DataFrame(
                inputs = rand([0, 1], 20),
                inputs_2 = rand([0, 1], 20),
                actions = rand([0, 1], 20),
                actions_2 = rand([0, 1], 20),
                ID = [
                    repeat(["A"], 5)
                    repeat(["B"], 5)
                    repeat(["A"], 5)
                    repeat(["B"], 5)
                ],
                category = [repeat(["X"], 10); repeat(["Y"], 10)],
            )
            #Create agent
            agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)
            #Set prior
            prior = Dict(
                "learning_rate" => LogitNormal(0.0, 1.0),
                "action_precision" => truncated(Normal(0.0, 1.0), lower = 0),
            )

            #Create model
            model = create_model(
                agent,
                prior,
                data,
                input_cols = :inputs,
                action_cols = :actions,
                grouping_cols = :ID,
            )
        end

        #Set samplings settings
        sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
        n_iterations = 10000
        n_chains = 10
        sampling_kwargs = (; progress = true)

        results = fit_model(
            model;
            parallelization = MCMCDistributed(),
            sampler = sampler,
            n_iterations = n_iterations,
            n_chains = n_chains,
            sampling_kwargs...,
        )
        rmprocs(workers())

        @test results isa ActionModels.FitModelResults
    end
end
