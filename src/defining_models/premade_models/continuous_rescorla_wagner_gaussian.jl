export ContinuousRescorlaWagnerGaussian

@Base.kwdef struct ContinuousRescorlaWagnerGaussian <: AbstractPremadeModel
    learning_rate::Float64 = 0.1
    action_noise::Float64 = 1
    initial_value::Float64 = 0
end


function ActionModel(config::ContinuousRescorlaWagnerGaussian)
    
    function continuous_rescorla_wagner_gaussian(attributes::ModelAttributes, observation::Float64)
        #Read in parameters and states
        parameters = load_parameters(attributes)
        states = load_states(attributes)

        learning_rate = parameters.learning_rate
        action_noise = parameters.action_noise
        previous_value = states.value

        ##We dont have any settings in this model. If we had, we would read them in as well.
        ##-----This is where the update step starts -------

        ##Get new value state
        new_value = previous_value + learning_rate * (observation - previous_value)
    
        ##-----This is where the update step ends -------
        ##Create Bernoulli normal distribution our action probability which we calculated in the update step
        action_distribution = Distributions.Normal(new_value, action_noise)
    
        ##Update the states and save them to agent's history
        update_state!(attributes, :value, new_value)
        update_state!(attributes, :observation, observation)
    
        ## return the action distribution to sample actions from
        return action_distribution
    end
    
    ## Create model 
    parameters = (
        learning_rate = Parameter(config.learning_rate),
        action_noise = Parameter(config.action_noise),
        initial_value = InitialStateParameter(config.initial_value, :value),
    )
    states = (
        value = State(Float64),
        observation = State(Float64),
    )

    observations = (;
        observation = Observation(Float64)
    )

    actions = (;
        report = Action(Normal)
    )

    return ActionModel(
        continuous_rescorla_wagner_gaussian,
        parameters = parameters,
        states = states,
        observations = observations,
        actions = actions,
    )

end