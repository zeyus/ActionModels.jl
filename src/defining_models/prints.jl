function Base.show(io::IO, ::MIME"text/plain", action_model::ActionModel)

    ## Build the output string
    output = IOBuffer()
    println(output, "-- ActionModel --")

    action_model_name = string(action_model.action_model)
    println(output, "Action model function: $action_model_name")

    n_parameters = length(action_model.parameters)
    println(output, "Number of parameters: $n_parameters")

    n_states = length(action_model.states)
    println(output, "Number of states: $n_states")

    if !isnothing(action_model.observations)
        n_observations = length(action_model.observations)
        println(output, "Number of observations: $n_observations")
    else
        println(output, "Observations not defined")
    end
    
    if !isnothing(action_model.actions)
        n_actions = length(action_model.actions)
        println(output, "Number of actions: $n_actions")
    else
        println(output, "Actions not defined")
    end

    # submodel info
    if !isnothing(action_model.submodel)
        submodel_type = string(typeof(action_model.submodel))
        println(output, "submodel type: $submodel_type")
    end

    ## Print the final string
    print(io, String(take!(output)))
end
