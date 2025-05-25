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
    formula::FormulaTerm
    prior::RegressionPrior
    inv_link::Function

    function Regression(
        formula::FormulaTerm,
        prior::RegressionPrior = RegressionPrior(),
        inv_link::Function = identity,
    )
        new(formula, prior, inv_link)
    end
    function Regression(
        formula::FormulaTerm,
        inv_link::Function,
        prior::RegressionPrior = RegressionPrior(),
    )
        new(formula, prior, inv_link)
    end
end

#########################################
### TYPES FOR MODEL FITTING AND TOOLS ###
#########################################
## Abstract type for containing fitting results for storing in the following ##
abstract type AbstractFittingResult end

### Structs for storing results of model fitting ###
Base.@kwdef struct ModelFitInfo
    session_ids::Vector{String}
    estimated_parameter_names::Tuple{Vararg{Symbol}}
end

Base.@kwdef mutable struct ModelFitResult
    chains::Chains
    session_parameters::Union{Nothing,AbstractFittingResult} = nothing
end

Base.@kwdef mutable struct ModelFit{T<:AbstractPopulationModel}
    model::DynamicPPL.Model
    population_model_type::T
    population_data::DataFrame
    info::ModelFitInfo
    prior::Union{ModelFitResult,Nothing} = nothing
    posterior::Union{ModelFitResult,Nothing} = nothing
end

### Structs for containing outputted session parameters and state trajectories ###
struct SessionParameters <: AbstractFittingResult
    value::NamedTuple{names,<:Tuple{Vararg{NamedTuple}}} where {names}
    modelfit::ModelFit
    estimated_parameter_names::Tuple{Vararg{Symbol}}
    session_ids::Vector{String}
    parameter_types::NamedTuple{
        parameter_names,
        <:Tuple{Vararg{Type}},
    } where {parameter_names}
    n_samples::Int
    n_chains::Int
end

struct StateTrajectories <: AbstractFittingResult
    value::NamedTuple{names,<:Tuple{Vararg{NamedTuple}}} where {names}
    modelfit::ModelFit
    state_names::Vector{Symbol}
    session_ids::Vector{String}
    state_types::NamedTuple{state_names,<:Tuple{Vararg{Type}}} where {state_names}
    n_samples::Int
    n_chains::Int
end


### Type for the save-resume functionality ###
@Base.kwdef struct SampleSaveResume
    save_every::Int = 100
    path::String = "./.samplingstate"
    chain_prefix::String = "ActionModels_chain_segment"
end

