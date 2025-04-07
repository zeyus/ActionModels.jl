
function sample_prior!(
    modelfit::ModelFit,
    n_samples::Integer = 1000,
    n_chains::Integer = 2,
    resample::Bool = false,
    parallelization::AbstractMCMC.AbstractMCMCEnsemble = MCMCSerial(),
)

    #If the prior has already been sampled
    if resample == false && !isnothing(modelfit.prior)
        #Do nothing
        return nothing
    end

    #Extract model
    model = modelfit.model

    #Sample prior
    chains = sample(model, Prior(), parallelization, n_samples, n_chains)

    #Store the prior
    modelfit.prior = chains
end