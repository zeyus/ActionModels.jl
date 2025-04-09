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
    n_parameters = length(modelfit.info.parameter_names)
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
