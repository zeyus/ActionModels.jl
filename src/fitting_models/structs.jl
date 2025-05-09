##########################################
### TYPES FOR SPECIFYING TURING MODELS ###
##########################################

#Structs for setting missing actions to be either skipped or inferred
abstract type AbstractMissingActions end
struct NoMissingActions <: AbstractMissingActions end
struct SkipMissingActions <: AbstractMissingActions end
struct InferMissingActions <: AbstractMissingActions end

#Abstract type for population models
abstract type AbstractPopulationModel end
#Type for custom population models
struct CustomPopulationModel <: AbstractPopulationModel end
struct RegressionPopulationModel <: AbstractPopulationModel end
struct IndependentPopulationModel <: AbstractPopulationModel end
struct SingleSessionPopulationModel <: AbstractPopulationModel end


## Types for the GLM population model ##
"""
Internal type for prepared regression priors
"""
Base.@kwdef mutable struct RegPrior
    β::Distribution
    σ::Union{Nothing,Vector{Distribution}}
end

"""
Input struct for setting regression priors
"""
Base.@kwdef struct RegressionPrior{D1<:Distribution,D2<:Distribution}
    β::Union{D1,Vector{D1}} = TDist(3)
    σ::Union{D2,Vector{Vector{D2}}} = truncated(TDist(3), lower = 0)
end

"""
Input struct for specifying a regression
"""
struct Regression
    formula::MixedModels.FormulaTerm
    prior::RegPrior
    inv_link::Function

    Regression(
        formula::MixedModels.FormulaTerm,
        prior::RegPrior = RegPrior(),
        inv_link::Function = identity,
    ) = begin
        new(formula, prior, inv_link)
    end
    Regression(
        formula::MixedModels.FormulaTerm,
        inv_link::Function,
        prior::RegPrior = RegPrior(),
    ) = begin
        new(formula, prior, inv_link)
    end
end

#########################################
### TYPES FOR MODEL FITTING AND TOOLS ###
#########################################
### Structs for containing outputted session parameters and state trajectories ###
struct SessionParameters
    estimated_parameter_names::Tuple{Vararg{Symbol}}
    session_ids::Vector{String}
    value::NamedTuple{parameter_names, <:Tuple{Vararg{NamedTuple}}} where parameter_names
    n_samples::Int
    n_chains::Int
end

struct StateTrajectories{T}
    state_names::Vector{Symbol}
    session_ids::Vector{String}
    value::Vector{
        AxisArray{
            Union{Missing,T},
            4,
            Array{Union{Missing,T},4},
            Tuple{
                Axis{:timestep,UnitRange{Int64}},
                Axis{:state,Vector{Symbol}},
                Axis{:sample,UnitRange{Int64}},
                Axis{:chain,UnitRange{Int64}},
            },
        },
    }
end

### Structs for storing results of model fitting ###
Base.@kwdef struct ModelFitInfo
    session_ids::Vector{String}
    estimated_parameter_names::Tuple{Vararg{Symbol}}
end

Base.@kwdef mutable struct ModelFitResult
    chains::Chains
    session_parameters::Union{Nothing,SessionParameters} = nothing
end

Base.@kwdef mutable struct ModelFit{T<:AbstractPopulationModel}
    model::DynamicPPL.Model
    population_model_type::T
    info::ModelFitInfo
    prior::Union{ModelFitResult,Nothing} = nothing
    posterior::Union{ModelFitResult,Nothing} = nothing
end


### Type for the save-resume functionality ###
struct SampleSaveResume
    save_every::Int
    path::String
    plot_progress::Bool
    chain_prefix::String
end
SampleSaveResume(;
    save_every::Int = 100,
    path = "./.samplingstate",
    plot_progress::Bool = false,
    chain_prefix = "ActionModels_chain_link",
) = SampleSaveResume(save_every, path, plot_progress, chain_prefix)

