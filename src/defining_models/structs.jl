############################################
### TYPE FOR INITIALIZING AN ACTIONMODEL ###
############################################

#For creating parameters
abstract type AbstractParameter end

struct Parameter{T<:Real} <: AbstractParameter 
    value::T
    type::Type{T}

    function Parameter(value::T) where {T<:Real}
        new{T}(value, T)
    end
    function Parameter(value::T, ::Type{T}) where {T<:Real}
        new{T}(value, T)
    end
    function Parameter(::Type{T}, value::T) where {T<:Real}
        new{T}(value, T)
    end
end

struct InitialStateParameter{T<:Real} <: AbstractParameter
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

#For creating states
abstract type AbstractState end
struct State{T} <: AbstractState
    value::T
    type::Type{T}

    function State(value::T) where {T}
        new{T}(value, T)
    end

    function State(value, ::Type{T}) where {T}
        new{T}(value, T)
    end

    function State( ::Type{T}, value) where {T}
        new{T}(value, T)
    end

    function State(::Type{T}) where {T}
        new{Union{T, Missing}}(missing, Union{T, Missing})
    end
end

#Supertype for substructs
abstract type AbstractSubstruct end

#ActionModel struct
struct ActionModel{T<:Union{AbstractSubstruct, Nothing}} 
    action_model::Function
    parameters::NamedTuple{parameter_names, <:Tuple{Vararg{<:AbstractParameter}}} where {parameter_names}
    states::NamedTuple{state_names, <:Tuple{Vararg{<:AbstractState}}} where {state_names}
    input_types::Union{Nothing,Tuple{Vararg{DataType}}}
    action_types::Union{Nothing,Tuple{Vararg{DataType}}}
    action_dist_types::Union{Nothing,Tuple{Vararg{DataType}}}
    substruct::T

    function ActionModel(
        action_model::Function,
        parameters::NamedTuple{parameter_names, <:Tuple{Vararg{<:AbstractParameter}}} where {parameter_names},
        states::NamedTuple{state_names, <:Tuple{Vararg{<:AbstractState}}} where {state_names};
        input_types::Union{Nothing,Tuple{Vararg{DataType}}} = nothing,
        action_types::Union{Nothing,Tuple{Vararg{DataType}}} = nothing,
        action_dist_types::Union{Nothing,Tuple{Vararg{DataType}}} = nothing,
        substruct::T = nothing
    ) where {T<:Union{AbstractSubstruct, Nothing}}
    
        #Check that action types are subtypes of real
        if !isnothing(action_types) && !all(action_types .<: Real)
            throw(ArgumentError("Not all specified action types $action_types are subtypes of Real."))
        end

        #Check that all action distribution types are subtypes of Distribution
        if !isnothing(action_dist_types) && !all(action_dist_types .<: Distribution)
            throw(ArgumentError("Not all specified action distribution types $action_dist_types are subtypes of Distribution."))
        end

        #Check initial state parameters
        for (parameter_name, parameter) in pairs(parameters)
            if parameter isa InitialStateParameter
                #Check that the parameter sets a state that exists
                state_name = parameter.state_name
                if !(state_name in keys(states))
                    throw(ArgumentError("The initial state parameter $parameter_name sets the state $state_name, but this state has not been specified by the user."))
                end
                #Check that the type of the parameter matches the type of the state
                state = states[state_name]
                if parameter.type <: state.type
                    throw(ArgumentError("The initial state parameter $parameter_name has type $(parameter.type), but sets the state $state_name which has type $(state.type)."))
                end
            end
        end

    return new{T}(action_model, parameters, states, input_types, action_types, action_dist_types, substruct)
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