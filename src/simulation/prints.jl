function Base.show(io::IO, ::MIME"text/plain", agent::Agent)

    ## Get information from agent struct
    action_model_name = string(agent.action_model)
    n_parameters = length(get_parameters(agent))
    n_states = length(get_states(agent))
    n_observations = length(agent.history[:action]) - 1

    ## Build the output string
    output = IOBuffer()
    println(output, "-- Agent struct --")
    println(output, "Action model name: $action_model_name")

    # submodel info
    if !isnothing(agent.submodel)
        submodel_type = string(typeof(agent.submodel))
        println(output, "submodel type: $submodel_type")
    end

    # Parameters
    if n_parameters > 0
        println(output, "Number of parameters: $n_parameters")
    end

    # States
    println(output, "Number of states (including the action): $n_states")

    # Number of observations
    if n_observations > 0
        println(output, "This agent has received $n_observations inputs")
    end

    ## Print the final string
    print(io, String(take!(output)))
end
