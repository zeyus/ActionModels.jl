### RESET ###
function reset!(agent::Agent)

    #Reset the model attributes
    reset!(agent.model_attributes)

    #Empty the history
    empty!.(agent.history)

    #Set initial states in history
    for (state_name, state_history) in pairs(agent.history)
        push!(
            state_history,
            returned_value(agent.model_attributes.initial_states[state_name]),
        )
    end
end


### SET ATTRIBUTES ###
function set_parameters!(
    agent::Agent,
    parameters::NamedTuple{parameter_keys,<:Tuple{Vararg{Real}}} where {parameter_keys},
)
    set_parameters!(agent.model_attributes, keys(parameter_names), values(parameters))
end
function set_states!(
    agent::Agent,
    states::NamedTuple{state_keys,<:Tuple{Vararg{Any}}} where {state_keys},
)
    set_states!(agent.model_attributes, keys(state_names), values(states))
end
function set_actions!(
    agent::Agent,
    actions::NamedTuple{action_keys,<:Tuple{Vararg{Real}}} where {action_keys},
)
    for (action_name, action_value) in actions
        #Set the action to the value
        agent.model_attributes.actions[action_name].value = action_value
    end
end


### GET ATTRIBUTES ###
## Getting all attributes ##
function get_parameters(agent::Agent)
    return get_parameters(agent.model_attributes)
end
function get_states(agent::Agent)
    return get_states(agent.model_attributes)
end
function get_actions(agent::Agent)
    return get_actions(agent.model_attributes)
end

## Getting single attributes ##
function get_parameters(agent::Agent, target_param::Symbol)
    return agent.model_attributes.parameters[target_param].value
end
function get_states(agent::Agent, target_state::Symbol)
    return agent.model_attributes.states[target_state].value
end
function get_actions(agent::Agent, target_action::Symbol)
    return agent.model_attributes.actions[target_action].value
end

## Getting multiple attributes ##
function get_parameters(agent::Agent, target_parameters::Vector{Symbol})
    return NamedTuple(
        parameter_name => get_parameters(agent, parameter_name) for
        parameter_name in target_parameters
    )
end
function get_states(agent::Agent, target_states::Vector{Symbol})
    return NamedTuple(
        state_name => get_states(agent, state_name) for state_name in target_states
    )
end
function get_actions(agent::Agent, target_actions::Vector{Symbol})
    return NamedTuple(
        action_name => get_actions(agent, action_name) for action_name in target_actions
    )
end

## Getting history ##
function get_history(agent::Agent)
    return agent.history
end
