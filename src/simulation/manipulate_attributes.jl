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
            get_states(agent.model_attributes, state_name),
        )
    end

    #Reset the number of timesteps
    agent.n_timesteps.value = 0

    return nothing
end


### SET ATTRIBUTES ###
## Setting multiple attributes with two tuples ##
function set_parameters!(
    agent::Agent,
    parameter_names::Tuple{Vararg{Symbol}},
    parameter_values::Tuple{Vararg{Union{R, AbstractArray{R}}}} where {R<:Real},
)
    set_parameters!(agent.model_attributes, parameter_names, parameter_values)
end
function set_states!(
    agent::Agent,
    state_names::Tuple{Vararg{Symbol}},
    state_values::Tuple{Vararg{Any}},
)
    set_states!(agent.model_attributes, state_names, state_values)
end
function set_actions!(
    agent::Agent,
    action_names::Tuple{Vararg{Symbol}},
    action_values::Tuple{Vararg{Real}},
)
    set_actions!(agent.model_attributes, action_name, action_value)
end
## Setting multiple attributes with a namedtuple ##
function set_parameters!(
    agent::Agent,
    parameters::NamedTuple{parameter_keys,<:Tuple{Vararg{Union{R,AbstractArray{R}}}}} where {R<:Real, parameter_keys},
)
    set_parameters!(agent.model_attributes, keys(parameters), values(parameters))
end
function set_states!(
    agent::Agent,
    states::NamedTuple{state_keys,<:Tuple{Vararg{Any}}} where {state_keys},
)
    set_states!(agent.model_attributes, keys(states), values(states))
end
function set_actions!(
    agent::Agent,
    actions::NamedTuple{action_keys,<:Tuple{Vararg{Real}}} where {action_keys},
)
    for (action_name, action_value) in actions
        store_action!(agent.model_attributes, action_name, action_value)
    end
end
## Setting multiple attributes with a vector and a tuple ##
function set_parameters!(
    agent::Agent,
    parameter_names::Vector{Symbol},
    parameter_values::Tuple{Vararg{Union{R, AbstractArray{R}}}} where {R<:Real},
)
    set_parameters!(agent.model_attributes, Tuple(parameter_names), parameter_values)
end
function set_states!(
    agent::Agent,
    state_names::Vector{Symbol},
    state_values::Tuple{Vararg{Any}},
)
    set_states!(agent.model_attributes, Tuple(state_names), state_values)
end
function set_actions!(
    agent::Agent,
    action_names::Vector{Symbol},
    action_values::Tuple{Vararg{Real}},
)
    for (action_name, action_value) in zip(action_names, action_values)
        store_action!(agent.model_attributes, action_name, action_value)
    end
end
## Setting single attributes ##
function set_parameters!(
    agent::Agent,
    target_param::Symbol,
    target_value::Union{R, AbstractArray{R}} where {R<:Real},
)
    set_parameters!(agent.model_attributes, target_param, target_value)
end
function set_states!(
    agent::Agent,
    target_state::Symbol,
    target_value::Any,
)
    set_states!(agent.model_attributes, target_state, target_value)
end
function set_actions!(
    agent::Agent,
    target_action::Symbol,
    target_value::Real,
)
    set_actions!(agent.model_attributes, target_action, target_value)
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
    return get_parameters(agent.model_attributes, target_param)
end
function get_states(agent::Agent, target_state::Symbol)
    return get_states(agent.model_attributes, target_state)
end
function get_actions(agent::Agent, target_action::Symbol)
    return get_actions(agent.model_attributes, target_action)
end

## Getting multiple attributes ##
function get_parameters(agent::Agent, target_parameters::Tuple{Vararg{Symbol}})
    return get_parameters(agent.model_attributes, target_parameters)
end
function get_states(agent::Agent, target_states::Tuple{Vararg{Symbol}})
    return get_states(agent.model_attributes, target_states)
end
function get_actions(agent::Agent, target_actions::Tuple{Vararg{Symbol}})
    return get_actions(agent.model_attributes, target_actions)
end

## Getting history ##
function get_history(agent::Agent)
    return agent.history
end
function get_history(agent::Agent, target_state::Tuple{Vararg{Symbol}})
    return get_history(agent.history, target_state)
end
function get_history(agent::Agent, target_state::Symbol)
    return agent.history[target_state]
end
