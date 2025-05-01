############################################
### TYPE FOR INITIALIZING AN ACTIONMODEL ###
############################################

## For creating parameters ##
abstract type AbstractParameter end

mutable struct Parameter{T<:Real} <: AbstractParameter
    value::T
    type::Type{T}

    function Parameter(value::T) where {T<:Real}
        new{Float64}(value, Float64)
    end
    function Parameter(value::T, ::Type{T}) where {T<:Real}
        new{T}(value, T)
    end
    function Parameter(::Type{T}, value::T) where {T<:Real}
        new{T}(value, T)
    end
end

mutable struct InitialStateParameter{T<:Real} <: AbstractParameter
    state::Symbol
    value::T
    type::Type{T}
    function InitialStateParameter(value::T, state_name::Symbol) where {T<:Real}
        new{T}(state_name, value, T)
    end
    function InitialStateParameter(value::T, state_name::Symbol, ::Type{T}) where {T<:Real}
        new{T}(state_name, value, T)
    end
    function InitialStateParameter(value::T, ::Type{T}, state_name::Symbol) where {T<:Real}
        new{T}(state_name, value, T)
    end
    function InitialStateParameter(state_name::Symbol, ::Type{T}, value::T) where {T<:Real}
        new{T}(state_name, value, T)
    end
    function InitialStateParameter(::Type{T}, value::T, state_name::Symbol) where {T<:Real}
        new{T}(state_name, value, T)
    end
    function InitialStateParameter(::Type{T}, state_name::Symbol, value::T) where {T<:Real}
        new{T}(state_name, value, T)
    end
end

## For creating states ##
abstract type AbstractState end
mutable struct State{T} <: AbstractState
    initial_value::Union{Missing,T}
    type::Type{T}

    function State(value::T) where {T}
        new{T}(value, T)
    end
    function State(value, ::Type{T}) where {T}
        new{T}(value, T)
    end
    function State(::Type{T}, value) where {T}
        new{T}(value, T)
    end
    function State(::Type{T}) where {T}
        new{T}(missing, T)
    end
end

## For creating observations ##
abstract type AbstractObservation end
struct Observation{T} <: AbstractObservation
    type::Type{T}

    function Observation(type::Type{T}) where {T}
        new{T}(T)
    end
end


## For creating actions ##
abstract type AbstractAction end
struct Action{T,TD} <: AbstractAction
    type::Type{T}
    distribution_type::Type{TD}

    function Action(
        action_type::Type{T},
        distribution_type::Type{TD},
    ) where {T<:Real,TD<:Distribution}

        if T != get_action_type(TD)
            @warn "Action type $T is different from the the action type $(get_action_type(TD)) which should match the distribution type $TD. Check that everything is in order"
        end
        new{T,TD}(action_type, distribution_type)
    end

    function Action(distribution_type::Type{TD}) where {TD<:Distribution}

        action_type = get_action_type(TD)

        T = action_type

        new{T,TD}(action_type, distribution_type)
    end

    function Action(action_type::Type{T}) where {T<:Real}
        new{T,Nothing}(action_type, Nothing)
    end
end

function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Univariate,Continuous}}
    return Float64
end
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Multivariate,Continuous}}
    return Array{Float64}
end
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Univariate,Discrete}}
    return Int64
end
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Multivariate,Discrete}}
    return Array{Int64}
end
function get_action_type(action_dist_type::Type{T}) where {T<:Distribution}
    return Nothing
end


## Supertype for submodels ##
abstract type AbstractSubmodel end

## ActionModel struct ##
abstract type AbstractActionModel end
struct ActionModel{T<:Union{AbstractSubmodel,Nothing}} <: AbstractActionModel
    action_model::Function
    parameters::NamedTuple{
        parameter_names,
        <:Tuple{Vararg{AbstractParameter}},
    } where {parameter_names}
    states::NamedTuple{state_names,<:Tuple{Vararg{AbstractState}}} where {state_names}
    observations::NamedTuple{
        observation_names,
        <:Tuple{Vararg{AbstractObservation}},
    } where {observation_names}
    actions::NamedTuple{action_names,<:Tuple{Vararg{AbstractAction}}} where {action_names}
    submodel::T

    function ActionModel(
        action_model::Function;
        parameters::Union{
            AbstractParameter,
            NamedTuple{parameter_names,<:Tuple{Vararg{AbstractParameter}}},
        } where {parameter_names},
        states::Union{
            AbstractState,
            NamedTuple{state_names,<:Tuple{Vararg{AbstractState}}},
        } where {state_names} = (;),
        observations::Union{
            AbstractObservation,
            NamedTuple{observation_names,<:Tuple{Vararg{AbstractObservation}}},
        } where {observation_names},
        actions::Union{
            AbstractAction,
            NamedTuple{action_names,<:Tuple{Vararg{AbstractAction}}},
        } where {action_names},
        submodel::T = nothing,
        verbose::Bool = true,
    ) where {T<:Union{AbstractSubmodel,Nothing}}

        #Make single structs into NamedTuples
        if parameters isa AbstractParameter
            parameters = (; parameter = parameters)
            if verbose
                @warn "A single parameter was passed, and is given the name :parameter. Use (; parameter_name = parameter) to specify the name of the parameter."
            end
        end
        if states isa AbstractState
            states = (; state = states)
            if verbose
                @warn "A single state was passed, and is given the name :state. Use (; state_name = state) to specify the name of the state."
            end
        end
        if actions isa AbstractAction
            actions = (; action = actions)
            if verbose
                @warn "A single action was passed, and is given the name :action. Use (; action_name = action) to specify the name of the action."
            end
        end
        if observations isa AbstractObservation
            observations = (; observation = observations)
            if verbose
                @warn "A single observation was passed, and is given the name :observation. Use (; observation_name = observation) to specify the name of the observation."
            end
        end

        #Check initial state parameters
        for (parameter_name, parameter) in pairs(parameters)
            if parameter isa InitialStateParameter
                #Check that the parameter sets a state that exists
                state_name = parameter.state
                if !(state_name in keys(states))
                    throw(
                        ArgumentError(
                            "The initial state parameter $parameter_name sets the state $state_name, but this state has not been specified by the user.",
                        ),
                    )
                end
                #Check that the type of the parameter matches the type of the state
                state = states[state_name]
                if !(parameter.type <: state.type)
                    throw(
                        ArgumentError(
                            "The initial state parameter $parameter_name has type $(parameter.type), but sets the state $state_name which has type $(state.type).",
                        ),
                    )
                end
            end
        end

        return new{T}(action_model, parameters, states, observations, actions, submodel)
    end
end




###################
### OTHER TYPES ###
###################
"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end

abstract type AbstractPremadeModel end