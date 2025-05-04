function get_session_parameters!(
    modelfit::ModelFit,
    prior_or_posterior::Symbol = :posterior;
    verbose::Bool = true,
)

    #Extract appropriate sample result
    if prior_or_posterior == :posterior

        if verbose && isnothing(modelfit.posterior)
            @warn "Posterior has not yet been sampled. Sampling now with default settings. Use sample_posterior! to choose other settings."
        end

        sample_posterior!(modelfit)

        sample_result = modelfit.posterior

    elseif prior_or_posterior == :prior

        if verbose && isnothing(modelfit.prior)
            @warn "sampling from the prior..."
        end

        sample_prior!(modelfit)

        sample_result = modelfit.prior
        
    else
        @error "use only either :posterior or :prior as the second argument"
    end

    #If the session_parameters have already been extracted, return them
    if !isnothing(sample_result.session_parameters)
        return sample_result.session_parameters
    end

    #Extract model & chains
    model = modelfit.model
    chains = sample_result.chains

    #Extract the session parameters (sample x chain array of vectors with parameter tuples for each session)
    returned_values = returned(model.args.population_model, chains)

    #Extract info
    estimated_parameter_names = modelfit.info.estimated_parameter_names
    n_parameters = length(estimated_parameter_names)
    session_ids = modelfit.info.session_ids
    n_sessions = length(session_ids)
    n_samples, n_chains = size(returned_values)

    #Create an empty AxisArray
    session_parameters =
        Array{Float64}(undef, n_sessions, n_parameters, n_samples, n_chains)
    session_parameters = AxisArray(
        session_parameters,
        Axis{:session}(session_ids),
        Axis{:parameter}(collect(estimated_parameter_names)),
        Axis{:sample}(1:n_samples),
        Axis{:chain}(1:n_chains),
    )

    #Go through each sampled parameter
    @progress for sample_idx = 1:n_samples
        for chain_idx = 1:n_chains

            sample_parameters = collect(returned_values[sample_idx, chain_idx])

            for session_idx = 1:n_sessions

                parameters = sample_parameters[session_idx]

                for parameter_idx = 1:n_parameters

                    parameter_value = parameters[parameter_idx]

                    #And set it in the appropriate place
                    session_parameters[session_idx, parameter_idx, sample_idx, chain_idx] =
                        parameter_value
                end
            end
        end
    end

    #Store as SessionParameters struct
    session_parameters = SessionParameters(
        estimated_parameter_names,
        session_ids,
        session_parameters,
    )

    #Save the session parameters
    sample_result.session_parameters = session_parameters

    return session_parameters
end
