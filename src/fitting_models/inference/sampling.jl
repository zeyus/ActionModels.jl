"""
    sample_posterior!(modelfit::ModelFit, parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial(); verbose=true, resample=false, save_resume=nothing, init_params=:sample_prior, n_samples=1000, n_chains=2, adtype=AutoForwardDiff(), sampler=NUTS(; adtype=adtype), sampler_kwargs...)

Sample from the posterior distribution of a fitted model using MCMC.

This function runs MCMC sampling for the provided `ModelFit` object, storing the results in `modelfit.posterior`. It supports saving and resuming sampling, parallelization, various ways of initializing parameters for the sampling, and specifying detailed settings for the sampling. Returns the sampled chains.

# Arguments
- `modelfit::ModelFit`: The model structure to sample with.
- `parallelization::AbstractMCMC.AbstractMCMCEnsemble`: Parallelization strategy (default: `MCMCSerial()`).
- `verbose::Bool`: Whether to display warnings (default: `true`).
- `resample::Bool`: Whether to force resampling even if results exist (default: `false`).
- `save_resume::Union{SampleSaveResume,Nothing}`: Save/resume configuration (default: `nothing`).
- `init_params::Union{Nothing,Symbol,Vector{Float64}}`: How to initialize the sampler (default: `:sample_prior`).
- `n_samples::Integer`: Number of samples per chain (default: `1000`).
- `n_chains::Integer`: Number of MCMC chains (default: `2`).
- `adtype`: Automatic differentiation type (default: `AutoForwardDiff()`).
- `sampler`: Sampler algorithm (default: `NUTS`).
- `sampler_kwargs...`: Additional keyword arguments for the sampler.

# Returns
- `Chains`: The sampled posterior chains.

# Examples
```jldoctest; setup = :(using ActionModels, DataFrames; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4,], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = (; learning_rate = LogitNormal()))

julia> model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id);

julia> chns = sample_posterior!(model, n_samples = 100, n_chains = 1);

julia> chns isa Chains
true

julia> chns = sample_posterior!(modelfit, MCMCSerial(); save_resume=SampleSaveResume(save_every = 50), n_samples = 100, n_chains = 1)

julia> chns isa Chains
true
```
"""
function sample_posterior!(
    modelfit::ModelFit,
    parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial();
    verbose::Bool = true,
    resample::Bool = false,
    save_resume::Union{SampleSaveResume,Nothing} = nothing,
    init_params::Union{Nothing,Symbol,Vector{Float64}} = :sample_prior,
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
            @warn """
            The gradients calculated with the chosen autodifferentiation type does not agree with ForwardDiff (atol: 1e-6).
            This may indicate a problem with the model. Take appropriate care.
            Gradients calculated with $(sampler.adtype) are: $(gradients)
            Gradients calculated with ForwardDiff are: $(forwarddiff_gradients)
            Differences are: $(gradient_differences)
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


"""
    sample_prior!(modelfit::ModelFit, parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial(); resample=false, n_samples=1000, n_chains=2)

Sample from the prior distribution of a fitted model using MCMC.

This function samples from the prior for the provided `ModelFit` object, storing the results in `modelfit.prior`. Returns the sampled chains.

# Arguments
- `modelfit::ModelFit`: The model structure to sample with.
- `parallelization::AbstractMCMC.AbstractMCMCEnsemble`: Parallelization strategy (default: `MCMCSerial()`).
- `resample::Bool`: Whether to force resampling even if results exist (default: `false`).
- `n_samples::Integer`: Number of samples per chain (default: `1000`).
- `n_chains::Integer`: Number of MCMC chains (default: `2`).

# Returns
- `Chains`: The sampled prior chains.

# Example
```jldoctest; setup = :(using ActionModels, DataFrames; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4,], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = (; learning_rate = LogitNormal()))

julia> model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id);

julia> chns = sample_prior!(model);

julia> chns isa Chains
true
```
"""
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
