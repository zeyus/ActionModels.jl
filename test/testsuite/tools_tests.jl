using Test

using ActionModels

ActionModels_path = dirname(dirname(pathof(ActionModels)))
test_path = joinpath(ActionModels_path, "test")

@testset "tools tests" begin
    # List the julia filenames in the tools testsuite
    tools_tests_filenames = glob("*.jl", joinpath(test_path, "testsuite", "tools"))

    for filename in tools_tests_filenames
        #Run it
        include(filename)
    end
end
