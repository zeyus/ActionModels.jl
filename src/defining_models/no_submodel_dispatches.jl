## This file contains the dispatches for the default NoSubModel type ##
## Initialze attributes ##
function initialize_attributes(
    submodel::NoSubModel,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}
    return NoSubModel()
end
## Reset ##
function reset!(submodel::NoSubModel)
    return nothing
end
## Getting all attributes ##
function get_parameters(submodel::NoSubModel) 
    return (;)
end
function get_states(submodel::NoSubModel)
    return (;)
end
## Getting a single attribute ##
function get_parameters(submodel::NoSubModel, target_param::Symbol)
    return AttributeError()
end
function get_states(submodel::NoSubModel, target_state::Symbol)
    return AttributeError()
end
## Getting multiple attributes ##
function get_parameters(submodel::NoSubModel, target_parameters::Tuple{Vararg{Symbol}})
    return AttributeError()
end
function get_states(submodel::NoSubModel, target_states::Tuple{Vararg{Symbol}})
    return AttributeError()
end

#Setting a single attribute
function set_parameters!(
    submodel::NoSubModel,
    target_param::Symbol,
    target_value::Any,
)
    return AttributeError()
end
function set_states!(
    submodel::NoSubModel,
    target_state::Symbol,
    target_value::Any,
)
    return AttributeError()
end
#Setting multiple attributes
function set_parameters!(
    submodel::NoSubModel,
    target_parameters::Tuple{Vararg{Symbol}},
    target_values::Tuple{Vararg{Real}},
)
    return AttributeError()
end
function set_states!(
    submodel::NoSubModel,
    target_states::Tuple{Vararg{Symbol}},
    target_values::Tuple{Vararg{Any}},
)
    return AttributeError()
end
