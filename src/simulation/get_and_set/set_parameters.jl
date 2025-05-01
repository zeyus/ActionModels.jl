###Function for setting a single parameter ###
"""
    set_parameters!(agent::Agent, target_param::Union{Symbol,Tuple}, param_value::Any)

Setting a single parameter value for an agent.

    set_parameters!(agent::Agent, parameter_values::Dict)

Set mutliple parameters values for an agent. Takes a dictionary of parameter names and values.
"""
function set_parameters! end

### Function for setting a single parameter ###
function set_parameters!(agent::Agent, target_param::Symbol, param_value::R) where R<:Real

    #If the parameter exists in the agent's parameters
    if target_param in keys(agent.parameters)
        #Set it
        agent.parameters[target_param] = param_value

        #If the parameter exists in the agent's initial state parameters
    elseif target_param in keys(agent.initial_state_parameters) 
        #Set it (it's not necessary to set it in the intial_states)
        agent.initial_state_parameters[target_param].value = param_value
    else
        #Otherwise, look in the submodel
        set_parameters!(agent.submodel, target_param, param_value)
    end
end

function set_parameters!(
    submodel::Nothing,
    target_param::Symbol,
    param_value::R,
) where R<:Real
    throw(
        ArgumentError("The specified parameter $target_param does not exist in the agent"),
    )
end


### Function for setting multiple parameters with a dict ###
function set_parameters!(agent::Agent, parameter_values::Dict{Symbol, R}) where R<:Real

    #For each parameter to set
    for (param_key, param_value) in parameter_values
        #Set that parameter
        set_parameters!(agent, param_key, param_value)
    end
end

### Function for setting multiple parameters with a vector of parmaeter names, and a vector, subarray or tuple of parameter values ###
function set_parameters!(agent::Agent, parameter_names::Vector{Symbol}, parameter_values::P) where {R<:Real, P<:Union{Vector{R}, NTuple{N,R} where N}}

    #For each parameter to set
    for (param_key, param_value) in zip(parameter_names, parameter_values)
        #Set that parameter
        set_parameters!(agent, param_key, param_value)
    end
end




#     #If the target param is a shared parameter
# elseif target_param in keys(agent.parameter_groups)

#     #Extract shared parameter
#     parameter_group = agent.parameter_groups[target_param]

#     #Set the shared parameter value
#     setfield!(parameter_group, :value, param_value)

#     #For each derived parameter
#     for grouped_parameter in parameter_group.grouped_parameters

#         #Set that parameter
#         set_parameters!(agent, grouped_parameter, param_value)
#     end