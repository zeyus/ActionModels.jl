using ActionModels
using Test
using Glob, Distributed

#Get the root path
ActionModels_path = dirname(dirname(pathof(ActionModels)))

@testset "all tests" begin

    test_path = ActionModels_path * "/test/"

    @testset "Aqua.jl tests" begin
        using Aqua
        Aqua.test_all(
            ActionModels,
            ambiguities = false,
            unbound_args = false, #TODO: turn these on again
        )
    end

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
