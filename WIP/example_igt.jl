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

#CHECK:
#- Mooncake has trouble fitting the reward reward_sensitivity


function bounded_exp(; lower = nothing, upper = 1e200)

    function _bounded_exp(x::T; 
        lower = if isnothing(lower) eps(T) else lower end, 
        upper = upper
        ) where {T<:Real}

        return max(
            min(
                exp(x), upper),
                lower
            )
    end

    return _bounded_exp
end


function bounded_logistic(; lower = nothing, upper = nothing)

    function _bounded_logistic(x::T; 
        lower = if isnothing(lower) eps(T)   else lower end, 
        upper = if isnothing(upper) 1-eps(T) else upper end 
        ) where {T<:Real}

        return max(
            min(
                logistic(x), upper),
                lower
            )
    end

    return _bounded_logistic
end



@testset "IGT example" begin
    # Example analysis using data from ahn et al 2014
    # Iowa Gambling Task on healthy controls and participants with heroin or amphetamine addictions
    # More details in docs/example_data/ahn_et_al_2014/ReadMe.txt

    # Import data
    data_healthy = CSV.read(joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_healthy_control.txt"), DataFrame)
    data_healthy[!, :clinical_group] .= "healthy"
    data_heroin = CSV.read(joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_heroin.txt"), DataFrame)
    data_heroin[!, :clinical_group] .= "heroin"
    data_amphetamine = CSV.read(joinpath(docs_path, "example_data/ahn_et_al_2014/IGTdata_amphetamine.txt"), DataFrame)
    data_amphetamine[!, :clinical_group] .= "amphetamine"

    #Combine into one dataframe
    ahn_data = vcat(data_healthy, data_heroin, data_amphetamine)
    ahn_data[!, :subjID] = string.(ahn_data[!, :subjID])

    # Make clumn wit total reward
    ahn_data[!, :reward] = Float64.(ahn_data[!, :gain] + ahn_data[!, :loss]);

    if true
        #subset the ahndata to have two subjID in each clinical_group
        ahn_data = filter(row -> row[:subjID] in ["103", "104", "337", "344",], ahn_data)
    end

    @testset "pvl-delta on igt" begin

        # create pvl-delta agent
        function pvl_delta(agent::Agent, input::Tuple{Int64, Float64})
            deck, reward = input

            learning_rate = agent.parameters["learning_rate"]
            reward_sensitivity = agent.parameters["reward_sensitivity"]
            loss_aversion = agent.parameters["loss_aversion"]
            inv_temperature = agent.parameters["inv_temperature"]

            expected_value = agent.states["expected_value"]
 
            action_probabilities = softmax(ActionModels.ad_val.(expected_value) * ActionModels.ad_val(inv_temperature))

            if reward >= 0
                prediction_error = (reward ^ reward_sensitivity) - expected_value[deck]
            else
                prediction_error = -loss_aversion * (abs(reward) ^ reward_sensitivity) - expected_value[deck]
            end

            expected_value[deck] = expected_value[deck] + learning_rate * prediction_error
            
            update_states!(agent, "expected_value", expected_value)

            return Categorical(action_probabilities)
        end

        agent = init_agent(pvl_delta,
                        parameters = Dict(  "learning_rate" => 0.05,
                                            "reward_sensitivity" => 0.4,
                                            "inv_temperature" => 1.3,
                                            "loss_aversion" => 0.5),
                        states = Dict("expected_value" => zeros(Real, 4)
                        ))                

        #Create model
        model = create_model(agent,
                            [
                                #@formula(learning_rate ~ clinical_group + (1|subjID)),
                                #@formula(inv_temperature ~ clinical_group + (1|subjID)),
                                @formula(reward_sensitivity ~ clinical_group + (1|subjID))
                                #@formula(loss_aversion ~ clinical_group + (1|subjID))
                                ],
                            ahn_data,
                            priors = [
                                #RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
                                #RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
                                RegressionPrior(β = TDist(1)*100, σ = truncated(TDist(1)*100, lower = 0)), #RegressionPrior(β = Normal(0, 1), σ = Exponential(1))
                                #RegressionPrior(β = Normal(0, 1), σ = Exponential(1)),
                            ],
                            inv_links = Function[
                                #logistic,
                                #bounded_exp(),
                                logistic,
                                #bounded_exp(),
                            ],
                            action_cols = [:deck],
                            input_cols = [:deck, :reward],
                            grouping_cols = [:subjID])

        # AD = AutoForwardDiff()
        # AD = AutoReverseDiff(; compile = false)
        # AD = AutoReverseDiff(; compile = true) #Explodes memory
        import Mooncake; AD = AutoMooncake(; config = nothing)
        # AD = AutoZygote()
        # using ADTypes: AutoEnzyme; using Enzyme; AD = AutoEnzyme(); 

        #Set samplings settings
        sampler = NUTS(-1, 0.65; adtype = AD) 
        n_iterations = 50
        sampling_kwargs = (; progress = true)

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)


    
        for i in 1:100
            print(i)
            map_estimate = maximum_a_posteriori(model, adtype=AD)
        end

        
        samples = sample(model, sampler, n_iterations; initial_params=map_estimate.values.array, sampling_kwargs...)

        

        #### INDEPENDENT AGENTS ###
        model = create_model(agent,
                            Dict(
                                "reward_sensitivity" => LogitNormal(0, 1),
                                "learning_rate" => LogitNormal(0, 1),
                                "inv_temperature" => LogNormal(0, 1),
                                "loss_aversion" => LogNormal(0, 1),
                            ),
                            ahn_data,
                            action_cols = [:deck],
                            input_cols = [:deck, :reward],
                            grouping_cols = [:subjID])



        ### SINGLE AGENT ###
        single_data = filter(row -> row[:subjID] in ["103"], ahn_data)
        inputs = Matrix(single_data[!, [:deck, :reward]])
        actions = single_data[!, :deck]

        model = create_model(agent,
                            Dict(
                                "reward_sensitivity" => LogitNormal(0, 1),
                                "learning_rate" => LogitNormal(0, 1),
                                "inv_temperature" => LogNormal(0, 1),
                                "loss_aversion" => LogNormal(0, 1),
                            ),
                            inputs,
                            actions)


        agent_parameters = extract_quantities(model, samples)
        estimates_df = get_estimates(agent_parameters, DataFrame)
        
        using StatsPlots
        plot(samples)


        h5open("pvldelta_mooncake.h5", "w") do f
            write(f, samples)
          end

        samples = sample(model.args.population_model, sampler, n_iterations; sampling_kwargs...)

    end

    @testset "simple model on IGT" begin

        function categorical_random(agent::Agent, input::Tuple{Int64, Float64})

            deck, reward = input

            temperature = agent.parameters["temperature"]

            #Avoid overflow
            temperature = min(temperature, 1e10)
            temperature = max(temperature, 1e-10)
            
            if reward >= 0
                temperature = 1/temperature
            end

            #Do a softmax of the values
            action_probabilities = softmax([0.7, 0.1, 0.1, 0.1] * temperature)
    
            return Categorical(action_probabilities)
        end
        agent = init_agent(categorical_random,
                                parameters = Dict("temperature" => 1))  

        #Create model
        model = create_model(agent,
                            @formula(temperature ~ clinical_group + (1|subjID)),
                            # @formula(temperature ~ 1),
                            ahn_data,
                            # priors = RegressionPrior(β = [Normal(0, 0.1)]),
                            inv_links = exp,
                            action_cols = [:deck],
                            input_cols = [:deck, :reward],
                            grouping_cols = [:subjID])
                        
        # AD = AutoForwardDiff()
        # AD = AutoReverseDiff(; compile = false)
        # AD = AutoReverseDiff(; compile = true) #Seems to work! 
        import Mooncake; AD = AutoMooncake(; config = nothing) #Works with only main effects
        # AD = AutoZygote()

        #Set samplings settings
        sampler = NUTS(-1, 0.65; adtype = AD) 
        n_iterations = 1000
        sampling_kwargs = (; progress = true)

        samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    
    end
end
