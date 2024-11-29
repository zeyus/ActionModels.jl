
using Pkg
Pkg.activate("../../docs")

using Test
using LogExpFunctions

using ActionModels
using Distributions
using DataFrames
using MixedModels
using Turing
using CSV


@testset "pvl-delta on igt" begin
    # Example analysis using data from ahn et al 2014
    # Iowa Gambling Task on healthy controls and participants with heroin or amphetamine addictions
    # More details in docs/example_data/ahn_et_al_2014/ReadMe.txt

    # Import data
    data_healthy = CSV.read("../../docs/example_data/ahn_et_al_2014/IGTdata_healthy_control.txt", DataFrame)
    data_healthy[!, :clinical_group] .= "healthy"
    data_heroin = CSV.read("../../docs/example_data/ahn_et_al_2014/IGTdata_heroin.txt", DataFrame)
    data_heroin[!, :clinical_group] .= "heroin"
    data_amphetamine = CSV.read("../../docs/example_data/ahn_et_al_2014/IGTdata_amphetamine.txt", DataFrame)
    data_amphetamine[!, :clinical_group] .= "amphetamine"

    ahn_data = vcat(data_healthy, data_heroin, data_amphetamine)
    ahn_data[!, :subjID] = string.(ahn_data[!, :subjID])

    # model total reward
    ahn_data[!, :reward] = ahn_data[!, :gain] + ahn_data[!, :loss]

    # create pvl-delta agent
    function pvl_delta(agent::Agent, input::Tuple{Int64, Int64})
        deck, reward = input

        learning_rate = agent.parameters["learning_rate"]
        reward_sensitivity = agent.parameters["reward_sensitivity"]
        loss_aversion = agent.parameters["loss_aversion"]
        temperature = agent.parameters["temperature"]

        expected_value = agent.states["expected_value"]

        weighted_action_probabilities = ActionModels.ad_val(temperature) .* expected_value
        action_probabilities = exp.(weighted_action_probabilities) ./ sum(exp.(weighted_action_probabilities))

        if reward >= 0
            prediction_error = (reward ^ reward_sensitivity) - expected_value[deck]
        else
            prediction_error = -loss_aversion * (abs(reward) ^ reward_sensitivity) - expected_value[deck]
        end

        expected_value[deck] = expected_value[deck] + learning_rate * prediction_error
        update_states!(agent, "expected_value", expected_value)

        return Categorical(ActionModels.ad_val.(action_probabilities))
    end

    agent = init_agent(pvl_delta,
                       parameters = Dict("learning_rate" => 0.05,
                                         "reward_sensitivity" => 0.4,
                                         "temperature" => 1.3,
                                         "loss_aversion" => 0.5),
                       states = Dict("expected_value" => tzeros(Real, 4)))


    model = create_model(agent,
                         @formula(learning_rate ~ 1),
                         # @formula(temperature ~ 1),
                         ahn_data,
                         # priors = RegressionPrior(β = [Normal(0, 0.1)]),
                         inv_links = logistic,
                         action_cols = [:deck],
                         input_cols = [:deck, :reward],
                         grouping_cols = [:subjID])

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true)) # blows up memory
    # sampler = NUTS(-1, 0.65; adtype = AutoForwardDiff())
    n_iterations = 10
    sampling_kwargs = (; progress = false)

    samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    # samples = sample(model.args.population_model, sampler, n_iterations; sampling_kwargs...)



    # @testset "test another thing" begin

    #     # cateorical
    #     function simple(agent::Agent, input::Tuple{Int64, Int64})
    #         deck, reward = input

    #         temperature = agent.parameters["temperature"]
    #         value = agent.states["value"]

    #         weighted_action_probabilities = ActionModels.ad_val(temperature) .* value # Float64[10, 15, 2, 5]
    #         action_probabilities = exp.(weighted_action_probabilities) ./ sum(exp.(weighted_action_probabilities))

    #         update_states!(agent, "value", value .+ 1)
    #         return Categorical(ActionModels.ad_val.(action_probabilities))
    #     end

    #     agent = init_agent(simple,
    #                        parameters = Dict("temperature" => 1.3),
    #                        states = Dict("value" => Real[10, 15, 2, 5]))


    #     model = create_model(agent,
    #                          @formula(temperature ~ clinical_group),
    #                          ahn_data,
    #                          priors = RegressionPrior(β = [Normal(0, 0.1), Normal(0, 0.1), Normal(0, 0.1)]),
    #                          inv_links = exp,
    #                          action_cols = [:deck],
    #                          input_cols = [:deck, :reward],
    #                          grouping_cols = [:subjID])

    #     #Set samplings settings
    #     sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    #     n_iterations = 10
    #     sampling_kwargs = (; progress = false)

    #     samples = sample(model, sampler, n_iterations; sampling_kwargs...)
    #     # samples = sample(model.args.population_model, sampler, n_iterations; sampling_kwargs...)


    # end
end
