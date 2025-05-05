using Test
using ActionModels

@testset "simulation" begin
    ### SETUP ###
    action_model = ActionModel(ContinuousRescorlaWagnerGaussian())

    agent = init_agent(action_model)

    @testset "give_observations!" begin

        observations = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

        actions = simulate!(agent, observations)

        @test length(actions) == length(observations)
        
    end

    @testset "parameter and state API" begin
        #Variations of get_states
        get_states(agent)

        get_states(agent, :value)

        get_states(agent, [:value, :action])

        #Variations of get_parameters
        get_parameters(agent)

        get_parameters(agent, :initial_value)

        get_parameters(agent, [:initial_value, :learning_rate])

        #Variations of set_parameters
        set_parameters!(agent, :initial_value, 1)

        #Variations of get_history
        get_history(agent, :value)

        get_history(agent)

        reset!(agent)
    end
end
