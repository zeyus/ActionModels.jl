### RESET ###
function reset!(agent::Agent)

    #Reset the model attributes
    reset!(agent.model_attributes)

    #Set initial states in history
    for (state_name, state_history) in pairs(agent.history)
        #Empty the history
        empty!(state_history)
        #Add the initial state to the history
        push!(
            state_history,
            return_value(agent.model_attributes.initial_states[state_name]),
        )
    end
end


### SET ATTRIBUTES ###
## Setting multiple attributes with a namedtuple ##
function set_parameters!(
    agent::Agent,
    parameters::NamedTuple{parameter_keys,<:Tuple{Vararg{Real}}} where {parameter_keys},
)
    for (parameter_name, parameter_value) in parameters
        set_parameters!(agent, parameter_name, parameter_value)
    end
end
function set_states!(
    agent::Agent,
    states::NamedTuple{state_keys,<:Tuple{Vararg{Any}}} where {state_keys},
)
    for (state_name, state_value) in states
        set_states!(agent, state_name, state_value)
    end
end
function set_actions!(
    agent::Agent,
    actions::NamedTuple{action_keys,<:Tuple{Vararg{Real}}} where {action_keys},
)
    for (action_name, action_value) in actions
        set_actions!(agent, action_name, action_value)
    end
end

## Setting multiple attributes with two tuples ##
function set_parameters!(
    agent::Agent,
    parameter_names::Tuple{Vararg{Symbol}},
    parameter_values::Tuple{Vararg{Real}},
)
    for (parameter_name, parameter_value) in zip(parameter_names, parameter_values)
        set_parameters!(agent, parameter_name, parameter_value)
    end
end
function set_states!(
    agent::Agent,
    state_names::Tuple{Vararg{Symbol}},
    state_values::Tuple{Vararg{Any}},
)
    for (state_name, state_value) in zip(state_names, state_values)
        set_states!(agent, state_name, state_value)
    end
end
function set_actions!(
    agent::Agent,
    action_names::Tuple{Vararg{Symbol}},
    action_values::Tuple{Vararg{Real}},
)
    for (action_name, action_value) in zip(action_names, action_values)
        set_actions!(agent, action_name, action_value)
    end
end
## Setting multiple attributes with a vector and a tuple ##
function set_parameters!(
    agent::Agent,
    parameter_names::Vector{Symbol},
    parameter_values::Tuple{Vararg{Real}},
)
    for (parameter_name, parameter_value) in zip(parameter_names, parameter_values)
        set_parameters!(agent, parameter_name, parameter_value)
    end
end
function set_states!(
    agent::Agent,
    state_names::Vector{Symbol},
    state_values::Tuple{Vararg{Any}},
)
    for (state_name, state_value) in zip(state_names, state_values)
        set_states!(agent, state_name, state_value)
    end
end
function set_actions!(
    agent::Agent,
    action_names::Vector{Symbol},
    action_values::Tuple{Vararg{Real}},
)
    for (action_name, action_value) in zip(action_names, action_values)
        set_actions!(agent, action_name, action_value)
    end
end

## Setting single attributes ##
function set_parameters!(agent::Agent, target_param::Symbol, value::R) where {R<:Real}
    agent.model_attributes.parameters[target_param].value = value
end
function set_states!(agent::Agent, target_state::Symbol, value::T) where {T<:Any}
    agent.model_attributes.states[target_state].value = value
end
function set_actions!(agent::Agent, target_action::Symbol, value::R) where {R<:Real}
    agent.model_attributes.actions[target_action].value = value
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
function get_history(agent::Agent, target_state::Symbol)
    return agent.history[target_state]
end
