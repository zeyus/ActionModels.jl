export BinaryRescorlaWagnerSoftmax

Base.@kwdef struct BinaryRescorlaWagnerSoftmax <: AbstractPremadeModel
    learning_rate::Float64 = 0.1
    action_noise::Float64 = 1
    initial_value::Float64 = 0
end

function ActionModel(config::BinaryRescorlaWagnerSoftmax)

    #Create function
    function binary_rescorla_wagner_softmax(attributes::ModelAttributes, observation::Int64)

        #Read in parameters
        learning_rate = attributes.parameters.learning_rate
        action_precision = 1 / attributes.parameters.action_noise

        #Read in states
        old_value = attributes.states.value

        #Sigmoid transform the value
        old_value_probability = 1 / (1 + exp(-old_value))

        #Get new value state
        new_value = old_value + learning_rate * (observation - old_value_probability)

        #Pass through softmax to get action probability
        action_probability = 1 / (1 + exp(-action_precision * new_value))

        #Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
        action_distribution = Distributions.Bernoulli(action_probability)

        #Update states
        update_state!(attributes, :value, new_value)
        update_state!(attributes, :observation, observation)

        return action_distribution
    end

    ## Create model 
    parameters = (
        learning_rate = Parameter(config.learning_rate),
        action_noise = Parameter(config.action_noise),
        initial_value = InitialStateParameter(config.initial_value, :value),
    )
    states = (value = State(Float64), observation = State(Float64))
    observations = (; observation = Observation(Int64))
    actions = (; report = Action(Bernoulli))

    return ActionModel(
        binary_rescorla_wagner_softmax,
        parameters = parameters,
        states = states,
        observations = observations,
        actions = actions,
    )

end