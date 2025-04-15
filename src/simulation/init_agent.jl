function init_agent(
    action_model::ActionModel{T};
    save_history::Bool = true,
) where T

    ##Create action model struct
    agent = Agent(
        action_model = action_model.action_model,
        submodel = action_model.submodel,
        parameters = Dict(),
        initial_state_parameters = Dict(),
        initial_states = Dict(),
        states = Dict(),
        save_history = save_history,
    )

    ##Add parameters to either initial state parameters or parameters
    for (param_key, parameter) in pairs(action_model.parameters)
        
        #If the param is an initial state parameter
        if parameter isa InitialStateParameter

            #Add the parameter using the state as key
            agent.initial_state_parameters[param_key] = parameter
            agent.initial_states[parameter.state] = parameter

        else
            #For other parameters, add to parameters
            agent.parameters[param_key] = parameter.value
        end
    end

    ##Add states
    if !isnothing(action_model.states)
        for (state_key, state_value) in pairs(action_model.states)
            agent.states[state_key] = state_value.value
        end
    end

    #If an action state was not specified
    if !(:action in keys(agent.states))
        #Add an empty action state
        agent.states[:action] = missing
    end

    #Reset the submodel to make sure initial states are correct, after setting the grouped parameters
    reset!(agent.submodel)

    #Initialize states
    for (param_key, initial_state) in pairs(agent.initial_state_parameters)

        #Extract the state and value
        state_key = initial_state.state
        initial_value = initial_state.value

        #Set initial state
        agent.states[state_key] = initial_value
    end

    #For each specified state
    for (state_key, state_value) in agent.states
        #Add it to the history
        agent.history[state_key] = [state_value]
    end

    #Check agent for settings of shared parameters
    check_agent(agent)

    return agent
end


"""
Function for checking the structure of the agent
"""
function check_agent(agent::Agent)

    if length(agent.parameter_groups) > 0

        ## Check for the same derived parameter in multiple shared parameters 
        #Get out the derived parameters of all shared parameters 
        grouped_parameters = [
            parameter for list_of_grouped_parameters in [
                agent.parameter_groups[parameter_key].grouped_parameters for
                parameter_key in keys(agent.parameter_groups)
            ] for parameter in list_of_grouped_parameters
        ]

        #check for duplicate names
        if length(grouped_parameters) > length(unique(grouped_parameters))
            #Throw an error
            throw(
                ArgumentError(
                    "At least one parameter is set by multiple shared parameters. This is not supported.",
                ),
            )
        end
    end


end







# #If there is only one parameter group, wrap it in a vector
# if parameter_groups isa ParameterGroup
#     parameter_groups = [parameter_groups]
# end

# #Go through each specified shared parameter
# for parameter_group in parameter_groups

#     #check if the name of the shared parameter is part of its own derived parameters
#     if parameter_group.name in parameter_group.parameters
#         throw(
#             ArgumentError(
#                 "The shared parameter $parameter_group is among the parameters it is defined to set",
#             ),
#         )
#     end

#     #Set the parameter group in the agent
#     agent.shared_parameters[parameter_group.name] = GroupedParameters(
#         value = parameter_group.value,
#         grouped_parameters = parameter_group.parameters,
#     )

#     #Set the parameters 
#     set_parameters!(agent, parameter_group, parameter_group.value)

# end