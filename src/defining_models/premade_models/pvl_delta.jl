export PVLDelta

Base.@kwdef struct PVLDelta <: AbstractPremadeModel
    n_decks::Int64 = 4
    learning_rate::Float64 = 0.1
    action_precision::Float64 = 1
    reward_sensitivity::Float64 = 0.5
    loss_aversion::Float64 = 1
    initial_value::Array{Float64} = zeros(Float64, n_decks)
end

function ActionModel(config::PVLDelta)

    function pvl_delta(agent::Agent, input::Tuple{Int64,Float64})

        deck, reward = input

        learning_rate = agent.parameters[:learning_rate]
        reward_sensitivity = agent.parameters[:reward_sensitivity]
        loss_aversion = agent.parameters[:loss_aversion]
        action_precision = agent.parameters[:action_precision]

        expected_value = agent.states[:expected_value]

        #Get action probabilities by softmaxing expected values for each deck
        action_probabilities = softmax(expected_value * action_precision)

        #Avoid underflow and overflow
        action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
        action_probabilities = action_probabilities / sum(action_probabilities)

        #Calculate prediction error
        if reward >= 0
            prediction_error = (reward^reward_sensitivity) - expected_value[deck]
        else
            prediction_error =
                -loss_aversion * (abs(reward)^reward_sensitivity) - expected_value[deck]
        end

        #Update expected values
        new_expected_value = [
            expected_value[deck_idx] +
            learning_rate * prediction_error * (deck == deck_idx) for deck_idx = 1:n_decks
        ]

        update_states!(agent, :expected_value, new_expected_value)

        return Categorical(action_probabilities)
    end

    parameters = (
            learning_rate = Parameter(config.learning_rate),
            reward_sensitivity = Parameter(config.reward_sensitivity),
            action_precision = Parameter(config.action_precision),
            loss_aversion = Parameter(config.loss_aversion),
            initial_value = InitialStateParameter(config.initial_value, :expected_value),
        )

    states = (; expected_value = State(zeros(Float64, 4)))

    observations = (;
        deck = Observation(Int64),
        reward = Observation(Float64),
    )

    actions = (;
        choice = Action(Categorical),
    )

    return ActionModel(
        pvl_delta,
        parameters = parameters,
        states = states,
        observations = observations,
        actions = actions,
    )
end