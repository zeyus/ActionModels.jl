"""
    reset!(agent::Agent)

Reset the agent's model attributes and history to their initial states.

This function resets all model parameters, states, and actions to their initial values, clears the agent's history, and sets the timestep counter to zero. This is useful for running new simulations with the same agent instance.

# Arguments
- `agent::Agent`: The agent whose attributes and history will be reset.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> reset!(agent)
```
"""
function reset!(agent::Agent)

    #Reset the model attributes
    reset!(agent.model_attributes)

    #Set initial states in history
    for (state_name, state_history) in pairs(agent.history)
        #Empty the history
        empty!(state_history)
        #Add the initial state to the history
        push!(state_history, get_states(agent.model_attributes, state_name))
    end

    #Reset the number of timesteps
    agent.n_timesteps.value = 0

    return nothing
end


"""
    set_parameters!(agent::Agent, parameter_names::Tuple{Vararg{Symbol}}, parameter_values::Tuple{Vararg{Union{R,AbstractArray{R}}}})

Set multiple model parameters for an agent using tuples of names and values.

# Arguments
- `agent::Agent`: The agent whose parameters will be set.
- `parameter_names::Tuple{Vararg{Symbol}}`: Tuple of parameter names.
- `parameter_values::Tuple{Vararg{Union{R,AbstractArray{R}}}}`: Tuple of parameter values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_parameters!(agent, (:learning_rate, :action_noise), (0.2, 0.1))
```
"""
function set_parameters!(
    agent::Agent,
    parameter_names::Tuple{Vararg{Symbol}},
    parameter_values::Tuple{Vararg{Union{R,AbstractArray{R}}}} where {R<:Real},
)
    set_parameters!(agent.model_attributes, parameter_names, parameter_values)
end
"""
    set_states!(agent::Agent, state_names::Tuple{Vararg{Symbol}}, state_values::Tuple{Vararg{Any}})

Set multiple model states for an agent using tuples of names and values.

# Arguments
- `agent::Agent`: The agent whose states will be set.
- `state_names::Tuple{Vararg{Symbol}}`: Tuple of state names.
- `state_values::Tuple{Vararg{Any}}`: Tuple of state values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_states!(agent, (:expected_value,), (0.0,))
```
"""
function set_states!(
    agent::Agent,
    state_names::Tuple{Vararg{Symbol}},
    state_values::Tuple{Vararg{Any}},
)
    set_states!(agent.model_attributes, state_names, state_values)
end
"""
    set_actions!(agent::Agent, action_names::Tuple{Vararg{Symbol}}, action_values::Tuple{Vararg{Real}})

Set multiple actions as having been previously chosen for an agent using tuples of names and values.

# Arguments
- `agent::Agent`: The agent whose actions will be set.
- `action_names::Tuple{Vararg{Symbol}}`: Tuple of action names.
- `action_values::Tuple{Vararg{Real}}`: Tuple of action values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_actions!(agent, (:report,), (1,))
```
"""
function set_actions!(
    agent::Agent,
    action_names::Tuple{Vararg{Symbol}},
    action_values::Tuple{Vararg{Real}},
)
    set_actions!(agent.model_attributes, action_names, action_values)
end
## Setting multiple attributes with a namedtuple ##
"""
    set_parameters!(agent::Agent, parameters::NamedTuple)

Set multiple model parameters for an agent using a NamedTuple.

# Arguments
- `agent::Agent`: The agent whose parameters will be set.
- `parameters::NamedTuple`: NamedTuple of parameter names and values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_parameters!(agent, (learning_rate=0.2, action_noise=0.1))
```
"""
function set_parameters!(
    agent::Agent,
    parameters::NamedTuple{
        parameter_keys,
        <:Tuple{Vararg{Union{R,AbstractArray{R}}}},
    } where {R<:Real,parameter_keys},
)
    set_parameters!(agent.model_attributes, keys(parameters), values(parameters))
end
"""
    set_states!(agent::Agent, states::NamedTuple)

Set multiple model states for an agent using a NamedTuple.

# Arguments
- `agent::Agent`: The agent whose states will be set.
- `states::NamedTuple`: NamedTuple of state names and values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_states!(agent, (expected_value=0.0,))
```
"""
function set_states!(
    agent::Agent,
    states::NamedTuple{state_keys,<:Tuple{Vararg{Any}}} where {state_keys},
)
    set_states!(agent.model_attributes, keys(states), values(states))
end
"""
    set_actions!(agent::Agent, actions::NamedTuple)

Set multiple actions as having been previously chosen for an agent using a NamedTuple.

# Arguments
- `agent::Agent`: The agent whose actions will be set.
- `actions::NamedTuple`: NamedTuple of action names and values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_actions!(agent, (;report=1.,))
```
"""
function set_actions!(
    agent::Agent,
    actions::NamedTuple{action_keys,<:Tuple{Vararg{Real}}} where {action_keys},
)   
    set_actions!(agent.model_attributes, keys(actions), values(actions))
end
## Setting multiple attributes with a vector and a tuple ##
"""
    set_parameters!(agent::Agent, parameter_names::Vector{Symbol}, parameter_values::Tuple{Vararg{Union{R,AbstractArray{R}}}})

Set multiple model parameters for an agent using a vector of names and a tuple of values.

# Arguments
- `agent::Agent`: The agent whose parameters will be set.
- `parameter_names::Vector{Symbol}`: Vector of parameter names.
- `parameter_values::Tuple{Vararg{Union{R,AbstractArray{R}}}}`: Tuple of parameter values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_parameters!(agent, [:learning_rate, :action_noise], (0.2, 0.1))
```
"""
function set_parameters!(
    agent::Agent,
    parameter_names::Vector{Symbol},
    parameter_values::Tuple{Vararg{Union{R,AbstractArray{R}}}} where {R<:Real},
)
    set_parameters!(agent.model_attributes, Tuple(parameter_names), parameter_values)
end
"""
    set_states!(agent::Agent, state_names::Vector{Symbol}, state_values::Tuple{Vararg{Any}})

Set multiple model states for an agent using a vector of names and a tuple of values.

# Arguments
- `agent::Agent`: The agent whose states will be set.
- `state_names::Vector{Symbol}`: Vector of state names.
- `state_values::Tuple{Vararg{Any}}`: Tuple of state values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_states!(agent, [:expected_value], (0.0,))
```
"""
function set_states!(
    agent::Agent,
    state_names::Vector{Symbol},
    state_values::Tuple{Vararg{Any}},
)
    set_states!(agent.model_attributes, Tuple(state_names), state_values)
end
"""
    set_actions!(agent::Agent, action_names::Vector{Symbol}, action_values::Tuple{Vararg{Real}})

Set multiple action as previously selected for an agent using a vector of names and a tuple of values.

# Arguments
- `agent::Agent`: The agent whose actions will be set.
- `action_names::Vector{Symbol}`: Vector of action names.
- `action_values::Tuple{Vararg{Real}}`: Tuple of action values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_actions!(agent, [:report], (1,))
```
"""
function set_actions!(
    agent::Agent,
    action_names::Vector{Symbol},
    action_values::Tuple{Vararg{Real}},
)
    set_actions!(agent.model_attributes, Tuple(action_names), action_values)
end
## Setting single attributes ##
"""
    set_parameters!(agent::Agent, target_param::Symbol, target_value::Union{R,AbstractArray{R}})

Set a single model parameter for an agent.

# Arguments
- `agent::Agent`: The agent whose parameter will be set.
- `target_param::Symbol`: Name of the parameter.
- `target_value::Union{R,AbstractArray{R}}`: Value to set.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_parameters!(agent, :learning_rate, 0.2)
```
"""
function set_parameters!(
    agent::Agent,
    target_param::Symbol,
    target_value::Union{R,AbstractArray{R}} where {R<:Real},
)
    set_parameters!(agent.model_attributes, target_param, target_value)
end
"""
    set_states!(agent::Agent, target_state::Symbol, target_value::Any)

Set a single model state for an agent.

# Arguments
- `agent::Agent`: The agent whose state will be set.
- `target_state::Symbol`: Name of the state.
- `target_value::Any`: Value to set.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_states!(agent, :expected_value, 0.0)
```
"""
function set_states!(agent::Agent, target_state::Symbol, target_value::Any)
    set_states!(agent.model_attributes, target_state, target_value)
end
"""
    set_actions!(agent::Agent, target_action::Symbol, target_value::Real)

Set a single model action for an agent.

# Arguments
- `agent::Agent`: The agent whose action will be set.
- `target_action::Symbol`: Name of the action.
- `target_value::Real`: Value to set.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> set_actions!(agent, :report, 1)
```
"""
function set_actions!(agent::Agent, target_action::Symbol, target_value::Real)
    set_actions!(agent.model_attributes, target_action, target_value)
end


### GET ATTRIBUTES ###
## Getting all attributes ##
"""
    get_parameters(agent::Agent)

Get all model parameters for an agent.

# Arguments
- `agent::Agent`: The agent whose parameters will be retrieved.

# Returns
- NamedTuple of all parameter names and values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_parameters(agent)
(action_noise = 1.0, learning_rate = 0.1, initial_value = 0.0)
```
"""
function get_parameters(agent::Agent)
    return get_parameters(agent.model_attributes)
end
"""
    get_states(agent::Agent)

Get all model states for an agent.

# Arguments
- `agent::Agent`: The agent whose states will be retrieved.

# Returns
- NamedTuple of all state names and values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_states(agent)
(expected_value = 0.0,)
```
"""
function get_states(agent::Agent)
    return get_states(agent.model_attributes)
end
"""
    get_actions(agent::Agent)

Get all previously selected actions for an agent.

# Arguments
- `agent::Agent`: The agent whose actions will be retrieved.

# Returns
- NamedTuple of all action names and values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_actions(agent)
(report = missing,)
```
"""
function get_actions(agent::Agent)
    return get_actions(agent.model_attributes)
end

## Getting single attributes ##
"""
    get_parameters(agent::Agent, target_param::Symbol)

Get a single model parameter for an agent.

# Arguments
- `agent::Agent`: The agent whose parameter will be retrieved.
- `target_param::Symbol`: Name of the parameter.

# Returns
- Value of the specified parameter.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_parameters(agent, :learning_rate)
0.1
```
"""
function get_parameters(agent::Agent, target_param::Symbol)
    return get_parameters(agent.model_attributes, target_param)
end
"""
    get_states(agent::Agent, target_state::Symbol)

Get a single model state for an agent.

# Arguments
- `agent::Agent`: The agent whose state will be retrieved.
- `target_state::Symbol`: Name of the state.

# Returns
- Value of the specified state.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_states(agent, :expected_value)
0.0
```
"""
function get_states(agent::Agent, target_state::Symbol)
    return get_states(agent.model_attributes, target_state)
end
"""
    get_actions(agent::Agent, target_action::Symbol)

Get a single model action for an agent.

# Arguments
- `agent::Agent`: The agent whose action will be retrieved.
- `target_action::Symbol`: Name of the action.

# Returns
- Value of the specified action.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_actions(agent, :report)
missing

julia> observe!(agent, 0.5);

julia> action = get_actions(agent, :report);

julia> action isa Real
true
```
"""
function get_actions(agent::Agent, target_action::Symbol)
    return get_actions(agent.model_attributes, target_action)
end

## Getting multiple attributes ##
"""
    get_parameters(agent::Agent, target_parameters::Tuple{Vararg{Symbol}})

Get multiple model parameters for an agent using a tuple of names.

# Arguments
- `agent::Agent`: The agent whose parameters will be retrieved.
- `target_parameters::Tuple{Vararg{Symbol}}`: Tuple of parameter names.

# Returns
- Tuple of parameter values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_parameters(agent, (:learning_rate, :action_noise))
(learning_rate = 0.1, action_noise = 1.0)
```
"""
function get_parameters(agent::Agent, target_parameters::Tuple{Vararg{Symbol}})
    return get_parameters(agent.model_attributes, target_parameters)
end
"""
    get_states(agent::Agent, target_states::Tuple{Vararg{Symbol}})

Get multiple model states for an agent using a tuple of names.

# Arguments
- `agent::Agent`: The agent whose states will be retrieved.
- `target_states::Tuple{Vararg{Symbol}}`: Tuple of state names.

# Returns
- Tuple of state values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_states(agent, (:expected_value,))
(expected_value = 0.0,)
```
"""
function get_states(agent::Agent, target_states::Tuple{Vararg{Symbol}})
    return get_states(agent.model_attributes, target_states)
end
"""
    get_actions(agent::Agent, target_actions::Tuple{Vararg{Symbol}})

Get multiple previously selected actions for an agent using a tuple of names.

# Arguments
- `agent::Agent`: The agent whose actions will be retrieved.
- `target_actions::Tuple{Vararg{Symbol}}`: Tuple of action names.

# Returns
- Tuple of action values.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_actions(agent, (:report,))
(report = missing,)
```
"""
function get_actions(agent::Agent, target_actions::Tuple{Vararg{Symbol}})
    return get_actions(agent.model_attributes, target_actions)
end

## Getting history ##
"""
    get_history(agent::Agent)

Get the full state history for an agent.

# Arguments
- `agent::Agent`: The agent whose history will be retrieved.

# Returns
- Dictionary mapping state names to their history arrays.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_history(agent)
(expected_value = [0.0],)
```
"""
function get_history(agent::Agent)
    return agent.history
end
"""
    get_history(agent::Agent, target_state::Tuple{Vararg{Symbol}})

Get the history for multiple states for an agent.

# Arguments
- `agent::Agent`: The agent whose history will be retrieved.
- `target_state::Tuple{Vararg{Symbol}}`: Tuple of state names.

# Returns
- Tuple of history arrays for the specified states.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_history(agent, (:expected_value,))
(expected_value = [0.0],)
```
"""
function get_history(agent::Agent, target_state::Tuple{Vararg{Symbol}})
    return NamedTuple(state_name => getfield(agent.history, state_name) for state_name in target_state)
end
"""
    get_history(agent::Agent, target_state::Symbol)

Get the history for a single state for an agent.

# Arguments
- `agent::Agent`: The agent whose history will be retrieved.
- `target_state::Symbol`: Name of the state.

# Returns
- Array of state values over time.

# Example
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> get_history(agent, :expected_value)
1-element Vector{Float64}:
 0.0
```
"""
function get_history(agent::Agent, target_state::Symbol)
    return agent.history[target_state]
end
