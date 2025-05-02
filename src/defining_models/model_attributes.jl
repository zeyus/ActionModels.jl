#############################################
####### INITIALIZING MODEL ATTRIBUTES #######
#############################################

function initialize_attributes(
    action_model::ActionModel,
    initial_states::NamedTuple{initial_state_keys,<:Tuple{Vararg{Any}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {initial_state_keys,TF,TI}

    parameters = initialize_variables(action_model.parameters, TF, TI)

    states = initialize_variables(action_model.states, TF, TI)

    actions = initialize_variables(action_model.actions, TF, TI)

    initial_states = map(
        state ->
            state isa ParameterDependentState ? parameters[state.parameter_name] :
            state,
        initial_states,
    )

    #TODO: intiialize submodel's attributes

    return ModelAttributes(parameters, states, actions, initial_states)
end


######################################################################
####### FUNCTION FOR LOADING TYPE FROM THE TURING MODEL HEADER #######
######################################################################
## Intializing variables with correct types for an attribute set ##
function initialize_variables(
    parameters::NamedTuple{names,<:Tuple{Vararg{AbstractParameter}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}

    return map(
        parameter -> Variable{load_type(parameter.type, TF, TI)}(parameter.value),
        parameters,
    )
end
function initialize_variables(
    states::NamedTuple{names,<:Tuple{Vararg{AbstractState}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}
    return map(
        state ->
            Variable{Union{Missing,load_type(state.type, TF, TI)}}(state.initial_value),
        states,
    ) #TODO: only allow missing for states with missing initial values?
end
function initialize_variables(
    actions::NamedTuple{names,<:Tuple{Vararg{AbstractAction}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}
    return map(
        action -> Variable{Union{Missing,load_type(action.type, TF, TI)}}(missing),
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





#######################################################
####### FUNCTIONS FOR MANIPULATING ATTRIBUTES #########
#######################################################
## Function for resetting states to their initial values ##
function reset!(model_attributes::ModelAttributes)
    #Go through each state
    for (state, initial_state) in
        zip(model_attributes.states, model_attributes.initial_states)
        #If the state is a parameter dependent state, set it to the value of the parameter
        if initial_state isa Variable
            state.value = initial_state.value
        else
            state.value = initial_state
        end
    end
end

## Function for setting model parameters ##
function set_parameters!(
    model_attributes::ModelAttributes,
    parameter_names::Vector{Symbol},
    parameters::Tuple{Vararg{Real}},
)
    #For each parameter name and value
    for (parameter_name, parameter_value) in zip(parameter_names, parameters)
        #Set the parameter to the value
        model_attributes.parameters[parameter_name].value = parameter_value
    end
end

## Function for updating a state ##
function update_state!(
    model_attributes::ModelAttributes,
    state_name::Symbol,
    state_value::S,
) where {S}
    #Set the state to the value
    model_attributes.states[state_name].value = state_value
end

## Function for saving an action ##
#For multiple actions
function store_action!(
    model_attributes::ModelAttributes,
    sampled_actions::Tuple{Vararg{Real}},
)
    #Go through each sampled action and corresponding action variable
    for (action_variable, sampled_action) in zip(model_attributes.actions, sampled_actions)
        #Set the action to the value
        action_variable.value = sampled_action
    end
end

#For single action
function store_action!(model_attributes::ModelAttributes, sampled_action::A) where {A<:Real}
    #Set the action to the value
    first(model_attributes.actions).value = sampled_action
end
