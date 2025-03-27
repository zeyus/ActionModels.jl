### TODO:
# Make check in full_model that returned agents_parameters is same length as inputs/actions
# Benchmark the try-catch in full_model
# Figure out helper function for extracting generated quantities (track_states true/false)
# Fix typing in create_model and full_model to give concrete types
# Make rename_chains also deal with missing actions
# full workflow: model comparison (PSIS)
# use arraydist for multiple actions 
# consider using a submodel for the agent model
# consider: make parameter recovery that uses a single model, so that pmap is unnecessary
# append the generated quantities to the chain (can use or reconstruct from https://github.com/farr/MCMCChainsStorage.jl)
# make bounded log and logit functions whihc don't break at too high or low values

using ActionModels, DataFrames
using Test
using Glob

#Get the root path
ActionModels_path = dirname(dirname(pathof(ActionModels)))

@testset "all tests" begin

    test_path = ActionModels_path * "/test/"

    @testset "quick tests" begin
        # Test the quick tests that are used as pre-commit tests
        include(test_path * "quicktests.jl")
    end

    @testset "unit tests" begin

        # List the julia filenames in the testsuite
        filenames = glob("*.jl", test_path * "testsuite")

        # For each file
        for filename in filenames
            #Run it
            include(filename)
        end
    end

    @testset "documentation tests" begin

        #Set up path for the documentation folder
        documentation_path = joinpath(ActionModels_path, "docs", "julia_files")

        # List the julia filenames in the documentation source files folder
        filenames = [glob("*/*.jl", documentation_path); glob("*.jl", documentation_path)] 

        for filename in filenames
            @testset "$(splitpath(filename)[end])" begin
                include(filename)
            end
        end
    end
end
