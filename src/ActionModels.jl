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
using ProgressMeter, Distributed #For parameter recovery
using Logging #For hiding sample rejections
@reexport using Distributions #Make distributions available to the user
using Turing: DynamicPPL, AbstractMCMC, AutoForwardDiff, AutoReverseDiff, AutoMooncake

## For defining action models ##
export Agent, RejectParameters, update_states!
export InitialStateParameter, ParameterGroup
export init_agent, premade_agent

## For simulation ##
export give_inputs!, single_input!, reset!
export get_history, get_states, get_parameters
export set_parameters!, set_save_history! # set_states!()
export plot_trajectory, plot_trajectory!

## For fitting models ##
export RegressionPrior
export create_model
export fit_model, ChainSaveResume
export extract_quantities, get_estimates, get_trajectories
export parameter_recovery
export plot_parameters, plot_trajectories


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



#Types for agents and errors
include("structs.jl")

#Functions for creating agents
include("create_agent/init_agent.jl")
include("create_agent/create_premade_agent.jl")
include("create_agent/multiple_actions.jl")
include("create_agent/check_agent.jl")
#Functions for fitting agents to data
include("fitting/create_model.jl")
include("fitting/create_session_model.jl")
include("fitting/population_models/independent_population_model.jl")
include("fitting/population_models/single_session_population_model.jl")
include("fitting/population_models/regression_population_model.jl")
include("fitting/helper_functions/extract_quantities.jl")
include("fitting/helper_functions/get_estimates.jl")
include("fitting/helper_functions/get_trajectories.jl")
include("fitting/helper_functions/helper_functions.jl")
include("fitting/helper_functions/rename_chains.jl")
include("fitting/helper_functions/evert.jl")
include("fitting/fit_model.jl")
include("fitting/parameter_recovery.jl")

#Plotting functions for agents
include("plots/plot_trajectories.jl")
include("plots/plot_parameters.jl")
include("plots/plot_trajectory.jl")

#Utility functions for agents
include("utils/get_history.jl")
include("utils/get_parameters.jl")
include("utils/get_states.jl")
include("utils/give_inputs.jl")
include("utils/reset.jl")
include("utils/set_parameters.jl")
include("utils/warn_premade_defaults.jl")
include("utils/pretty_printing.jl")
include("utils/update_states.jl")
include("utils/set_save_history.jl")

#Premade agents
include("premade_models/binary_rescorla_wagner_softmax.jl")
include("premade_models/continuous_rescorla_wagner_gaussian.jl")
end
