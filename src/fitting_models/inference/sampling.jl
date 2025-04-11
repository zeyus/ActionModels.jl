function sample_posterior!(
    modelfit::ModelFit,
    parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial();
    #Whether to display warnings
    verbose::Bool = true,
    #Whether to resample the posterior
    resample::Bool = false,
    #Whether to use save_resume
    save_resume::Union{SampleSaveResume,Nothing} = nothing,
    #Which way to choose initial parameters for the sampler
    init_params::Union{Nothing,Symbol,Vector{Float64}} = :MAP,
    #Sampling configurations
    n_samples::Integer = 1000,
    n_chains::Integer = 2,
    adtype = AutoForwardDiff(),
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm} = NUTS(;
        adtype = adtype,
    ),
    sampler_kwargs...,
)

    #If the posterior has already been sampled
    if resample == false && !isnothing(modelfit.posterior)
        #Do nothing
        return modelfit.posterior.chains
    end

    #Extract model
    model = modelfit.model

    #If init_params is a symbol, extract it in the corresponding way
    if init_params isa Symbol

        #Use the Maximum a Posteriori (MAP) estimate as the initial parameters
        if init_params == :MAP
            init_params = maximum_a_posteriori(model, adtype = sampler.adtype).values.array

            #Use the Maximum Likelihood (MLE) estimate as the initial parameters
        elseif init_params == :MLE
            init_params = maximum_likelihood(model, adtype = sampler.adtype).values.array

            #Draw a single initial sample from the prior
        elseif init_params == :sample_prior
            init_params = DynamicPPL.VarInfo(model)[:]

        else
            throw(
                ArgumentError(
                    "if init_params is a symbol, it must be either: :MAP, :MLE or :sample_prior",
                ),
            )
        end
    end

    #Check whether a gradient can be calculated at init_params
    if !isnothing(init_params)
        gradients = LogDensityProblems.logdensity_and_gradient(
            LogDensityFunction(model; adtype = sampler.adtype),
            init_params,
        )[2]
        if verbose && (any(isinf.(gradients)) || any(isnan.(gradients)))
            @warn """
            The initial parameters for the sampler return NaN or Inf gradients. 
            The sampler will instead be initialized with Turing-default random values.
            """
            init_params = nothing
        end

        #TODO: should this happen also if init_params is nothing?
        #Check if gradient agrees with ForwardDiff
        forwarddiff_gradients = LogDensityProblems.logdensity_and_gradient(
            LogDensityFunction(model; adtype = AutoForwardDiff()),
            init_params,
        )[2]

        gradient_differences = gradients .- forwarddiff_gradients
        if verbose && !all(isapprox.(gradient_differences, 0.0, atol = 1e-6))

            @show gradients
            @show forwarddiff_gradients

            @warn """
            The gradients calculated with the chosen autodifferentiation type does not agree with ForwardDiff (atol: 1e-6).
            This may indicate a problem with the model. Take appropriate care.
            """
        end
    end

    #If save_resume is not activated
    if isnothing(save_resume)

        #Sample the posterior
        chains = sample(
            model,
            sampler,
            parallelization,
            n_samples,
            n_chains;
            init_params = init_params,
            sampler_kwargs...,
        )
    else

        #Sample with save_resume
        chains = sample_save_resume(
            model,
            save_resume,
            n_samples,
            n_chains,
            parallelization,
            sampler,
            init_params;
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