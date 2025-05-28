export PVLDelta

Base.@kwdef struct PVLDelta <: AbstractPremadeModel
    n_options::Int64
    learning_rate::Float64 = 0.1
    action_noise::Float64 = 1
    reward_sensitivity::Float64 = 0.5
    loss_aversion::Float64 = 1
    initial_value::Array{Float64} = zeros(Float64, n_options)
    act_before_update::Bool = false
end

function ActionModel(config::PVLDelta)

    if !config.act_before_update

        am_function = function pvl_delta(
            attributes::ModelAttributes,
            chosen_option::Int64,
            reward::Float64,
        )

            ## Read in parameters and states ##
            parameters = load_parameters(attributes)
            states = load_states(attributes)

            α = parameters.learning_rate
            A = parameters.reward_sensitivity
            w = parameters.loss_aversion
            β = 1 / parameters.action_noise

            Ev = states.expected_value


            ## Update expected values ##
            #Calculate prediction error
            if reward >= 0
                prediction_error = (reward^A) - Ev[chosen_option]
            else
                prediction_error = -w * (abs(reward)^A) - Ev[chosen_option]
            end

            #Calculate new expected value for the chosen option
            # Ev[chosen_option] = Ev[chosen_option] + α * prediction_error
            Ev = [
                Ev[option_idx] + α * prediction_error * (chosen_option == option_idx) for
                option_idx = 1:config.n_options
            ]


            #Update the expected value for next timestep
            update_state!(attributes, :expected_value, Ev)


            ## Get action probabilities ##
            #Softmax over expected values for each option
            action_probabilities = softmax(Ev * β)

            #Avoid underflow and overflow
            action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
            action_probabilities = action_probabilities / sum(action_probabilities)

            return Categorical(action_probabilities)
        end

    else

        am_function = function pvl_delta_act_before_update(
            attributes::ModelAttributes,
            chosen_option::Int64,
            reward::Float64,
        )

            ## Read in parameters and states ##
            parameters = load_parameters(attributes)
            states = load_states(attributes)

            α = parameters.learning_rate
            A = parameters.reward_sensitivity
            w = parameters.loss_aversion
            β = 1 / parameters.action_noise

            Ev = states.expected_value


            ## Get action probabilities ##
            #Softmax over expected values for each option
            action_probabilities = softmax(Ev * β)

            #Avoid underflow and overflow
            action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
            action_probabilities = action_probabilities / sum(action_probabilities)


            ## Update expected values ##
            #Calculate prediction error
            if reward >= 0
                prediction_error = (reward^A) - Ev[chosen_option]
            else
                prediction_error = -w * (abs(reward)^A) - Ev[chosen_option]
            end

            #Calculate new expected value for the chosen option
            #Ev[chosen_option] = Ev[chosen_option] + α * prediction_error
            Ev = [
                Ev[option_idx] + α * prediction_error * (chosen_option == option_idx) for
                option_idx = 1:config.n_options
            ]

            #Update the expected value for next timestep
            update_state!(attributes, :expected_value, Ev)

            return Categorical(action_probabilities)
        end
    end

    parameters = (
        learning_rate = Parameter(config.learning_rate),
        reward_sensitivity = Parameter(config.reward_sensitivity),
        action_noise = Parameter(config.action_noise),
        loss_aversion = Parameter(config.loss_aversion),
        initial_value = InitialStateParameter(config.initial_value, :expected_value),
    )

    states = (; expected_value = State(Array{Float64}))

    observations = (; chosen_option = Observation(Int64), reward = Observation())

    actions = (; choice = Action(Categorical),)

    return ActionModel(
        am_function,
        parameters = parameters,
        states = states,
        observations = observations,
        actions = actions,
    )
end
