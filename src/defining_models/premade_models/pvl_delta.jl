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

    function pvl_delta(attributes::ModelAttributes, deck::Int64, reward::Float64)
        
        #Read in parameters and states
        parameters = get_parameters(attributes)
        states = get_states(attributes)

        learning_rate = parameters.learning_rate
        reward_sensitivity = parameters.reward_sensitivity
        loss_aversion = parameters.loss_aversion
        action_precision = parameters.action_precision

        previous_value = states.value

        #Get action probabilities by softmaxing expected values for each deck
        action_probabilities = softmax(previous_value * action_precision)

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
        new_value = [
            previous_value[deck_idx] +
            learning_rate * prediction_error * (deck == deck_idx) for deck_idx = 1:n_decks
        ]

        update_state!(attributes, :value, new_value)

        return Categorical(action_probabilities)
    end

    parameters = (
            learning_rate = Parameter(config.learning_rate),
            reward_sensitivity = Parameter(config.reward_sensitivity),
            action_precision = Parameter(config.action_precision),
            loss_aversion = Parameter(config.loss_aversion),
            initial_value = InitialStateParameter(config.initial_value, :value),
        )

    states = (; value = State(config.initial_value))

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