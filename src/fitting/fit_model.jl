struct ChainSaveResume
    save_every::Int
    path::String
    plot_progress::Bool
end

ChainSaveResume() = ChainSaveResume(0, tempdir(), false)

#####################################################################
### CONVENIENCE FUNCTION FOR DOING FULL MODEL FITTING IN ONE LINE ###
#####################################################################
function fit_model(
    model::DynamicPPL.Model;
    parallelization::Union{Nothing,AbstractMCMC.AbstractMCMCEnsemble}=nothing,
    sampler::Union{DynamicPPL.AbstractSampler,Turing.Inference.InferenceAlgorithm}=NUTS(
        -1,
        0.65;
        adtype=AutoReverseDiff(),
    ),
    n_iterations::Integer=1000,
    n_chains=1,
    show_sample_rejections::Bool=false,
    show_progress::Bool=true,
    save_chains::Union{Nothing,String}=nothing,
    save_resume::ChainSaveResume=ChainSaveResume(),
    sampler_kwargs...,
)

    ## Set logger ##
    #If sample rejection warnings are to be shown
    if show_sample_rejections
        #Use a standard logger
        sampling_logger = Logging.SimpleLogger()
    else
        #Use a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
    end

    ## Fit model ##
    if !isnothing(parallelization) && n_chains > 1
        #With parallelization
        chains = Logging.with_logger(sampling_logger) do
            # see if it's a threads or distributed ensemble
            if parallelization isa AbstractMCMC.AbstractMCMCThreads
                sample(model, sampler, n_iterations, n_chains; progress=show_progress, sampler_kwargs...)
            else
                sample(model, sampler, n_iterations, n_chains=n_chains; progress=show_progress, sampler_kwargs...)
            end
        end
    elseif save_resume.save_every < 1
        chains = Logging.with_logger(sampling_logger) do
            sample(model, sampler, n_iterations, n_chains=n_chains; progress=show_progress, sampler_kwargs...)
        end
    end

    return FitModelResults(chains, model)
end
