using Test

using ActionModels

ActionModels_path = dirname(dirname(pathof(ActionModels)))
test_path = joinpath(ActionModels_path, "test")

@testset "premade models" begin

    for ad_type in
        ["AutoForwardDiff", "AutoReverseDiff", "AutoReverseDiff(true)", "AutoMooncake"]

        #Select appropriate AD backend
        if ad_type == "AutoForwardDiff"
            AD = AutoForwardDiff()
        elseif ad_type == "AutoReverseDiff"
            AD = AutoReverseDiff()
        elseif ad_type == "AutoReverseDiff(true)"
            AD = AutoReverseDiff(; compile = true)
        elseif ad_type == "AutoMooncake"
            AD = AutoMooncake(; config = nothing)
        end

        # List the julia filenames in the premade models testsuite
        premade_models_filenames = glob("*.jl", joinpath(test_path, "testsuite", "premade_model_testsuite"))

        for filename in premade_models_filenames
            #Run it
            include(filename)
        end
    end
end
