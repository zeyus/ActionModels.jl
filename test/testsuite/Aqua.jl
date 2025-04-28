@testset "Aqua.jl tests" begin
    using ActionModels
    using Aqua
    #Aqua.test_all(ActionModels, ambiguities = false)
    Aqua.test_all(
        ActionModels,
        ambiguities = false,
        unbound_args = false,
        undefined_exports = true,
    )
end


