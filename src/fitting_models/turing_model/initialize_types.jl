function initialize_attributes(
    action_model::ActionModel,
    initial_states::NamedTuple{initial_state_keys,<:Tuple},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {initial_state_keys,TF,TI}

    parameters = NamedTuple{keys(action_model.parameters),<:Tuple{Vararg{Variable}}}(
        initialize_variables(attribute_types.parameters, TF, TI),
    )
    states = NamedTuple{attribute_names.states,<:Tuple{Vararg{Variable}}}(
        initialize_variables(attribute_types.states, TF, TI),
    )
    actions = NamedTuple{attribute_names.actions,<:Tuple{Vararg{Variable}}}(
        initialize_variables(attribute_types.actions, TF, TI),
    )

    initial_states = NamedTuple{keys(action_model.states), <:Tuple}(
        map(
            state ->
                state isa ParameterDependentState ? parameters[state.parameter_name] :
                state.initial_value,
            initial_states,
        ),
    )

    #TODO: intiialize submodel's attributes

    return ModelAttributes(
        parameters,
        states,
        actions,
        initial_states,
    )
end








#####################################################################################################################################################
####### FUNCTION FOR LOADING TYPE FROM THE TURING MODEL HEADER, NECESSARY FOR FORWARDDIFF AND REVERSEDIFF FOR THE AUTODIFFERENTIATION BACKEND #######
#####################################################################################################################################################
## Intializing variables with correct types for an attribute set ##
function initialize_variables(
    parameters::Tuple{Vararg{AbstractParameter}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}
    return map(
        parameter -> Variable(parameter.value, load_type(parameter.type, TF, TI)),
        parameters,
    )
end

function initialize_variables(
    states::Tuple{Vararg{AbstractState}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    return map(
        state ->
            Variable(state.initial_value, Union{Missing,load_type(states.type, TF, TI)}),
        states,
    ) #TODO: only allow missing for states with missing initial values?
end

function initialize_variables(
    actions::Tuple{Vararg{AbstractAction}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    return map(
        action -> Variable(missing, Union{Missing,load_type(action.type, TF, TI)}),
        actions,
    )
end


## Load the correct type for a single attribute ##
#For returning an array type
function load_type(
    ::Type{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {ST,T<:Array{ST},TF,TI}

    NT = load_type(ST, TF, TI)

    return Array{NT}
end

#For returning a single type
function load_type(
    ::Type{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {T<:AbstractFloat,TF,TI}
    return TF
end
function load_type(
    ::Type{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {T<:Integer,TF,TI}
    return TI
end
function load_type(::Type{T}, ::Type{TF} = Float64, ::Type{TI} = Int64) where {T,TF,TI}
    return T
end

