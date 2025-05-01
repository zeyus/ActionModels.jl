######################
### PRINT MODELFIT ###
######################
function Base.show(io::IO, ::MIME"text/plain", modelfit::ModelFit{T}) where T<:AbstractPopulationModel

    #Make I/O buffer
    output = IOBuffer()

    println(output, "-- ModelFit object --")

    #Get name of action model
    action_model_name = string(modelfit.model.args.agent_model.action_model)

    println(output, "Action model: $action_model_name")

    #Get population model type
    population_model_type = modelfit.population_model_type
    if population_model_type == IndependentPopulationModel()
        population_model_type = "Independent sessions population model"
    elseif population_model_type == RegressionPopulationModel()
        population_model_type = "Linear regression population model"
    else
        population_model_type = "Custom population model"
    end

    println(output, "$population_model_type")

    #Get info
    n_parameters = length(modelfit.info.estimated_parameter_names)
    n_sessions = length(modelfit.info.session_ids)

    println(output, "$n_parameters estimated action model parameters, $n_sessions sessions")

    if isnothing(modelfit.posterior)
        println(output, "Posterior not sampled")
    else
        n_samples, _, n_chains = size(modelfit.posterior.chains) 

        println(output, "Posterior: $n_samples samples, $n_chains chains")
    end

    if isnothing(modelfit.prior)
        println(output, "Prior not sampled")
    else
        n_samples, _, n_chains = size(modelfit.prior.chains) 

        println(output, "Prior: $n_samples samples, $n_chains chains")
    end

    ## Print the final string
    print(io, String(take!(output)))
end


################################
### PRINT STATE TRAJECTORIES ###
################################
function Base.show(io::IO, ::MIME"text/plain", state_trajectories::StateTrajectories{T}) where T
    #Make I/O buffer
    output = IOBuffer()

    println(output, "-- State trajectories object --")

    #Extract n sessions
    n_sessions = length(state_trajectories.session_ids)
    #Extract n_samples and n_chains
    n_samples, n_chains = size(first(state_trajectories.value))[3:4]

    println(output, "$n_sessions sessions, $n_chains chains, $n_samples samples")

    state_names = state_trajectories.state_names

    println(output, "$(length(state_names)) estimated states:")

    for state_name in state_trajectories.state_names
        println(output, "   $state_name")
    end

    ## Print the final string
    print(io, String(take!(output)))
end

################################
### PRINT SESSION PARAMETERS ###
################################

function Base.show(io::IO, ::MIME"text/plain", session_parameters::SessionParameters)
    #Make I/O buffer
    output = IOBuffer()

    println(output, "-- Session parameters object --")

    #Extract n sessions
    n_sessions = length(session_parameters.session_ids)
    #Extract n_samples and n_chains
    n_samples, n_chains = size(session_parameters.value)[3:4]

    println(output, "$n_sessions sessions, $n_chains chains, $n_samples samples")

    estimated_parameter_names = session_parameters.estimated_parameter_names

    println(output, "$(length(estimated_parameter_names)) estimated parameters:")

    for parameter_name in session_parameters.estimated_parameter_names
        println(output, "   $parameter_name")
    end

    ## Print the final string
    print(io, String(take!(output)))
end



