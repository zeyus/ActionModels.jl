docs_path = joinpath(@__DIR__, "..", "..", "docs")
using Pkg
Pkg.activate(docs_path)

using Test
using LogExpFunctions

using ActionModels
using Distributions
using DataFrames, CSV
using MixedModels
using Turing






function bounded_exp(; lower = nothing, upper = 1e200)

    function _bounded_exp(x::T; 
        lower = if isnothing(lower) eps(T) else lower end, 
        upper = upper
        ) where {T<:Real}

        return clamp(
            exp(x),
            lower,
            upper
        )
    end

    return _bounded_exp
end


function bounded_logistic(; lower = nothing, upper = nothing)

    function _bounded_logistic(x::T; 
        lower = if isnothing(lower) eps(T)   else lower end, 
        upper = if isnothing(upper) 1-eps(T) else upper end 
        ) where {T<:Real}


        return clamp(
            logistic(x),
            lower,
            upper
        )
    end

    return _bounded_logistic
end

@testset "JGET example" begin

    #Data from https://github.com/nacemikus/jget-schizotypy
    #Trial-level data
    JGET_data = CSV.read(joinpath(docs_path, "example_data/JGET/JGET_data_trial_preprocessed.csv"), DataFrame)
    JGET_data = select(JGET_data, [:trials, :ID, :session, :outcome, :response, :confidence])

    #Subject-level data
    subject_data = CSV.read(joinpath(docs_path, "example_data/JGET/JGET_data_sub_preprocessed.csv"), DataFrame)
    subject_data = select(subject_data, [:ID, :session, :pdi_total, :Age, :Gender, :Education])

    #Join the data
    JGET_data = innerjoin(JGET_data, subject_data, on = [:ID, :session])

    #Make session into a categorical variable
    JGET_data.session = string.(JGET_data.session)

    #Remove rows with missing values in pdi_total
    JGET_data = filter(row -> !ismissing(row[:pdi_total]), JGET_data)

    #make the outcome column Float64
    JGET_data.outcome = Float64.(JGET_data.outcome)

    if true
        #subset the data
        JGET_data = filter(row -> row[:ID] in [20, 74] && row[:session] in ["1", "2", "3"], JGET_data)
    end

    @testset "rescorla wagner" begin
        
        # create resocrla wagner aget
        function rescorla_wagner(agent::Agent, input::T) where T<:Real

            learning_rate = agent.parameters["learning_rate"]
            action_noise = agent.parameters["action_noise"]

            expected_value = agent.states["expected_value"]

            action_probability = Normal(expected_value, action_noise)

            new_expected_value = expected_value + learning_rate * (input - expected_value)
            
            update_states!(agent, "expected_value", new_expected_value)

            return action_probability
        end

        agent = init_agent(rescorla_wagner,
                        parameters = Dict(  "learning_rate" => 0.05,
                                            "action_noise" => 1),
                        states = Dict("expected_value" => 0)
                        )   

        model = create_model(agent,
                            [
                            # @formula(learning_rate ~ pdi_total + session  + (1|ID)),
                            # @formula(action_noise ~ pdi_total + session  + (1|ID)),
                            @formula(learning_rate ~ 1),
                            #@formula(action_noise ~ 1),
                            ],
                        JGET_data,
                        priors = [
                            RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
                            #RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
                        ],
                        inv_links = Function[
                            logistic,
                            #bounded_exp(),
                        ],
                        action_cols = [:response],
                        input_cols = [:outcome],
                        grouping_cols = [:ID, :session])

        AD = AutoForwardDiff()
        # AD = AutoReverseDiff(; compile = false)
        # AD = AutoReverseDiff(; compile = true)
        # import Mooncake; AD = AutoMooncake(; config = nothing)
        # AD = AutoZygote()
        # using ADTypes: AutoEnzyme; using Enzyme; AD = AutoEnzyme(); 
        
        #Set samplings settings
        sampler = NUTS(-1, 0.65; adtype = AD) 
        n_iterations = 100
        sampling_kwargs = (; progress = true)

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)

        using StatsPlots
        plot(samples)

        agent_parameters = extract_quantities(model, samples)
        estimates_df = get_estimates(agent_parameters, DataFrame)

    end

    @testset "JGET HGF" begin

    end
end
