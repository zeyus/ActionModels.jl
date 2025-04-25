export ContinuousRescorlaWagnerGaussian

@Base.kwdef struct ContinuousRescorlaWagnerGaussian <: AbstractPremadeModel
    learning_rate::Float64 = 0.1
    action_noise::Float64 = 1
    initial_value::Float64 = 0
end


function ActionModel(config::ContinuousRescorlaWagnerGaussian)
    
    function continuous_rescorla_wagner_gaussian(agent::Agent, observation::Float64)

        ## Read in parameters from the agent
        learning_rate = agent.parameters[:learning_rate]
        action_noise = agent.parameters[:action_noise]
    
        ## Read in states with an initial value
        old_value = agent.states[:value]
    
        ##We dont have any settings in this model. If we had, we would read them in as well.
        ##-----This is where the update step starts -------

        ##Get new value state
        new_value = old_value + learning_rate * (observation - old_value)
    
        ##-----This is where the update step ends -------
        ##Create Bernoulli normal distribution our action probability which we calculated in the update step
        action_distribution = Distributions.Normal(new_value, action_noise)
    
        ##Update the states and save them to agent's history
        update_states!(agent, :value, new_value)
        update_states!(agent, :observation, observation)
    
        ## return the action distribution to sample actions from
        return action_distribution
    end
    
    ## Create model 
    parameters = (
        learning_rate = Parameter(config.learning_rate, Real),
        action_noise = Parameter(config.action_noise, Real),
        initial_value = InitialStateParameter(config.initial_value, :value, Real),
    )
    states = (
        value = State(Real),
        observation = State(Real),
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