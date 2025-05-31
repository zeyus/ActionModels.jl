using Test

using ActionModels

ActionModels_path = dirname(dirname(pathof(ActionModels)))
test_path = joinpath(ActionModels_path, "test")

@testset "premade models" begin

    for AD in [
        "ForwardDiff",
        "ReverseDiff",
        "ReverseDiff Compiled",
        "Mooncake",
        "Enzyme Forward",
        "Enzyme Reverse",
        "FiniteDifferences",
    ]

        #Select appropriate AD backend
        if AD == "ForwardDiff"
            ad_type = AutoForwardDiff()
        elseif AD == "ReverseDiff"
            ad_type = AutoReverseDiff()
        elseif AD == "ReverseDiff Compiled"
            ad_type = AutoReverseDiff(; compile = true)
        elseif AD == "Mooncake"
            ad_type = AutoMooncake(; config = nothing)
        elseif AD == "Enzyme Forward"
            ad_type = AutoEnzyme(; mode = set_runtime_activity(Forward, true))
        elseif AD == "Enzyme Reverse"
            ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true))
        elseif AD == "FiniteDifferences"
            ad_type = AutoFiniteDifferences(; fdm = central_fdm(5, 1))
        end

        # List the julia filenames in the premade models testsuite
        premade_models_filenames =
            glob("*.jl", joinpath(test_path, "testsuite", "premade_models"))

        for filename in premade_models_filenames
            #Run it
            include(filename)
        end
    end
end
