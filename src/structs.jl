##################
## AGENT STRUCT ##
##################
"""
Agent struct
"""
Base.@kwdef mutable struct Agent
    action_model::Function
    substruct::Any
    parameters::Dict = Dict()
    initial_state_parameters::Dict{String,Any} = Dict()
    parameter_groups::Dict = Dict()
    states::Dict{String,Any} = Dict("action" => missing)
    history::Dict{String,Vector{Any}} = Dict("action" => [missing])
    settings::Dict{String,Any} = Dict()
    save_history::Bool = true
end


######################################
## STRUCTS FOR CREATE AND FIT MODEL ##
######################################
mutable struct FitModelResults
    chains::Chains
    model::DynamicPPL.Model
end

#Structs for setting missing actions to be either skipped or inferred
struct SkipMissingActions end
struct InferMissingActions end

"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end

"""
Population model type
"""
abstract type AbstractPopulationModel end


####################################
## STRUCTS FOR SETTING PARAMETERS ##
####################################
"""
Type to use for specifying a paramter that sets a state's initial value
"""
Base.@kwdef mutable struct InitialStateParameter
    state::Any
end

"""
Type for specifying a group of parameters
"""
Base.@kwdef mutable struct ParameterGroup
    name::String
    parameters::Vector
    value::Real
end

"""
Type for shared parameters containing both the parameter value and a vector of parameter names that will share that value
"""
Base.@kwdef mutable struct GroupedParameters
    value::Real
    grouped_parameters::Vector
end

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
Base.@kwdef struct RegressionPrior{D1<:Distribution, D2<:Distribution}
    β::Union{D1, Vector{D1}} = TDist(3)
    σ::Union{D2, Vector{Vector{D2}}} = truncated(TDist(3), lower = 0)
end

"""
Input struct for specifying a regression
"""
struct Regression
    formula::MixedModels.FormulaTerm
    prior::RegPrior
    inv_link::Function

    Regression(formula::MixedModels.FormulaTerm, prior::RegPrior = RegPrior(), inv_link::Function = identity) = begin
        new(formula, prior, inv_link)
    end
    Regression(formula::MixedModels.FormulaTerm, inv_link::Function, prior::RegPrior = RegPrior()) = begin
        new(formula, prior, inv_link)
    end
end