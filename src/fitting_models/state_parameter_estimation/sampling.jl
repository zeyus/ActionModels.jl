function sample_posterior!(
    modelfit::ModelFit,
    parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial();
    #Whether to use save_resume
    save_resume::Union{SampleSaveResume,Nothing} = nothing,
    #Sampling configurations
    n_samples::Integer = 1000,
    n_chains::Integer = 2,
    ad_type = AutoForwardDiff(),
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm} = NUTS(;
        adtype = ad_type,
    ),
    sampler_kwargs...,
)

    #Extract model
    model = modelfit.model

    #If save_resume is not activated
    if isnothing(save_resume)

        #Sample the posterior
        chains =
            sample(model, sampler, parallelization, n_samples, n_chains; sampler_kwargs...)
    else

        #Sample with save_resume
        chains = sample_save_resume(
            model,
            save_resume,
            n_samples,
            n_chains,
            parallelization,
            sampler,
            sampler_kwargs...,
        )
    end

     #Store the posterior
     modelfit.posterior = ModelFitResult(; chains = chains)

    return chains
end


function sample_prior!(
    modelfit::ModelFit,
    parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial();
    resample::Bool = false,
    n_samples::Integer = 1000,
    n_chains::Integer = 2,
)

    #If the prior has already been sampled
    if resample == false && !isnothing(modelfit.prior)
        #Do nothing
        return modelfit.prior.chains
    end

    #Extract model
    model = modelfit.model

    #Sample prior
    chains = sample(model, Prior(), parallelization, n_samples, n_chains)

    #Store the prior
    modelfit.prior = ModelFitResult(; chains = chains)

    return chains
end