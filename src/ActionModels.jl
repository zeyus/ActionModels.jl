module ActionModels

#Load packages
using Reexport 
using Turing #For fitting models
using RecipesBase #For defining plots
using DataFrames #For the input format to create_model
using AxisArrays #For storing session parameters and state trajectories
using StatsModels, MixedModels, LogExpFunctions #For the GLM population Model
using ForwardDiff, ReverseDiff, Mooncake #AD types to be compatible with
using HDF5 #For the save_resume functionality
using ProgressLogging #For progress bars
using Distributed #For parameter recovery and other heavy functions
using Logging #For hiding sample rejections
@reexport using Distributions #Make distributions available to the user
using Turing: DynamicPPL, AbstractMCMC, AutoForwardDiff, AutoReverseDiff, AutoMooncake

## For defining action models ##
export Agent, RejectParameters, update_states!
export InitialState, ParameterGroup
export init_agent, premade_agent

## For simulation ##
export give_inputs!, single_input!, reset!
export get_history, get_states, get_parameters
export set_parameters!, set_save_history! # set_states!()
export plot_trajectory, plot_trajectory!

## For fitting models ##
export create_model, RegressionPrior
export sample_prior!, sample_posterior!, SampleSaveResume
export get_session_parameters!, get_state_trajectories!, summarize
export parameter_recovery
export plot_parameters, plot_trajectories
export bounded_exp, bounded_logistic
export @formula


## Load premade agents ##
function __init__()
    # Only if not precompiling
    if ccall(:jl_generating_output, Cint, ()) == 0
        premade_agents["binary_rescorla_wagner_softmax"] =
            premade_binary_rescorla_wagner_softmax
        premade_agents["continuous_rescorla_wagner_gaussian"] =
            premade_continuous_rescorla_wagner_gaussian
    end
end


## Constants for creating ids and names consistently ##
const id_separator = "."
const id_column_separator = ":"
const tuple_separator = "."




### Functions for model definition ###
include(joinpath("defining_models", "structs.jl"))
include(joinpath("defining_models", "init_agent.jl"))
include(joinpath("defining_models", "create_premade_model.jl"))
include(joinpath("defining_models", "update_states.jl"))
include(joinpath("defining_models", "pretty_print.jl"))

#Read in all premade models
for premade_model_file in readdir(joinpath("src", "defining_models", "premade_models"))
    if endswith(premade_model_file, ".jl")
        include(joinpath("defining_models", "premade_models", premade_model_file))
    end
end

### Functions for simulation ###
include(joinpath("simulation", "give_inputs.jl"))
include(joinpath("simulation", "reset.jl"))
include(joinpath("simulation", "get_and_set", "set_parameters.jl"))
include(joinpath("simulation", "get_and_set", "set_save_history.jl"))
include(joinpath("simulation", "get_and_set", "get_parameters.jl"))
include(joinpath("simulation", "get_and_set", "get_states.jl"))
include(joinpath("simulation", "get_and_set", "get_history.jl"))
include(joinpath("simulation", "plots", "plot_trajectory.jl"))

### Functions for fitting models ###
include(joinpath("fitting_models", "structs.jl"))
include(joinpath("fitting_models", "turing_model", "create_model.jl"))
include(joinpath("fitting_models", "turing_model", "create_session_model.jl"))
include(joinpath("fitting_models", "turing_model", "helper_functions.jl"))
include(joinpath("fitting_models", "turing_model", "population_models", "independent_population_model.jl"))
include(joinpath("fitting_models", "turing_model", "population_models", "glm_population_model.jl"))
include(joinpath("fitting_models", "turing_model", "population_models", "single_session_population_model.jl"))
include(joinpath("fitting_models", "tools", "sampling.jl"))
include(joinpath("fitting_models", "tools", "save_resume.jl"))
include(joinpath("fitting_models", "tools", "parameter_recovery.jl"))
include(joinpath("fitting_models", "extract_results", "get_session_parameters!.jl"))
include(joinpath("fitting_models", "extract_results", "get_state_trajectories.jl"))
include(joinpath("fitting_models", "extract_results", "summarize.jl"))
include(joinpath("fitting_models", "plots", "plot_parameters.jl"))
include(joinpath("fitting_models", "plots", "plot_trajectories.jl"))

end
