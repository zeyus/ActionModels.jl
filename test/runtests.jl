using ActionModels
using Test, Documenter
using Glob, Distributed

#Get the root path
ActionModels_path = dirname(dirname(pathof(ActionModels)))

@testset "all tests" begin

    test_path = ActionModels_path * "/test/"

    @testset "Aqua.jl tests" begin
        using Aqua
        Aqua.test_all(ActionModels, unbound_args = false)
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

    # Run the doctests
    DocMeta.setdocmeta!(ActionModels, :DocTestSetup, :(using ActionModels); recursive = true)
    doctest(ActionModels)

    @testset "documentation tests" begin

        # Set up path for the documentation folder
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
