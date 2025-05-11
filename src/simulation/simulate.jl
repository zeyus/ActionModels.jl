###################################
### MAKING A SINGLE OBSERVATION ###
###################################
function observe!(agent::Agent, observation::T) where {T<:Any}

    #Run the action model to get the action distribution
    action_distribution = agent.action_model(agent.model_attributes, observation...)

    #Sample actions
    action = sample_actions!(action_distribution)

    #Store the action
    store_action!(agent.model_attributes, action)

    #Save states in history
    for (state_name, state_value) in pairs(agent.history)
        push!(
            state_value,
            get_states(agent, state_name),
        )
    end

    #Count the timestep
    agent.n_timesteps.value += 1

    #Return the action
    return action
end

## Functions for sampling a single action ##
function sample_actions!(action::D) where {D<:Distributions.Distribution}
    #Sample an action from the action distribution
    return rand(action)
end
function sample_actions!(action::Tuple{Vararg{D}}) where {D<:Distributions.Distribution}
    return map(sample_actions!, action)
end


#####################################
### SIMULATING MULTIPLE TIMESTEPS ###
#####################################
## With pre-specified observations ##
#With a vector where each element is a single observation or a tuple of observations
function simulate!(agent::Agent, observations::AbstractVector{T}) where {T<:Any}
    
    #Simulate forward
    actions = [observe!(agent, observation) for observation in observations]

    #Return the actions
    return actions
end
#With a matrix where each row is a single observation
function simulate!(agent::Agent, observations::AbstractMatrix{T}) where {T<:Any}
    
    #Simulate forward
    actions = [observe!(agent, observation) for observation in eachrow(observations)]

    #Return the actions
    return actions
end

#TODO: with an environment

#TODO: with another agent