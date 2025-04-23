using Test

using ActionModels
using DataFrames
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake

@testset "create_model_tests" begin

    ### SETUP ###
    #Generate dataset
    data = DataFrame(
        inputs = repeat([1, 1, 1, 2, 2, 2], 6),
        actions = vcat(
            [0, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0.5, 0.8, 1, 1.5, 1.8],
            [0, 2, 0.5, 4, 5, 3],
            [0, 0.1, 0.15, 0.2, 0.25, 0.3],
            [0, 0.2, 0.4, 0.7, 1.0, 1.1],
            [0, 2, 0.5, 4, 5, 3],
        ),
        age = vcat(
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
        ),
        id = vcat(
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
        ),
        treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
    )

    #Add a second set of actions and inputs
    data.actions_2 = data.actions
    data.inputs_2 = data.inputs

    #Define input and action cols
    input_cols = [:inputs]
    action_cols = [:actions]
    grouping_cols = [:id, :treatment]

    #Create action_model
    action_model = ActionModel(ContinuousRescorlaWagnerGaussian())

    prior = Dict(
        :learning_rate => LogitNormal(),
        :action_noise => LogNormal(),
        :initial_value => Normal(),
    )

    #Go through each supported AD type
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

        #Set sampling arguments
        n_iterations = 50
        sampling_kwargs = (; progress = false)
        sampler = NUTS(-1, 0.65; adtype = AD)

        ### TESTING MODEL TYPES ###
        @testset "single session ($ad_type)" begin
            #Extract inputs and actions from data
            inputs = data[!, :inputs]
            actions = data[!, :actions]

            #Create model
            model = create_model(action_model, prior, inputs, actions)

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "simple statistical model ($ad_type)" begin
            #Create model
            model = create_model(
                action_model,
                prior,
                data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = [:id],
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "custom population model ($ad_type)" begin

            #Define population model
            @model function custom_population_model()
                #Sample parameters (same across sessions)
                control_learning_rate ~ LogitNormal(0.0, 1.0)
                control_action_noise ~ LogNormal(0.0, 1.0)
                control_initial_value ~ Normal(0.0, 1.0)
                treatment_learning_rate ~ LogitNormal(0.0, 1.0)
                treatment_action_noise ~ LogNormal(0.0, 1.0)
                treatment_initial_value ~ Normal(0.0, 1.0)

                #Put into vector
                control_params = [(
                        control_learning_rate,
                        control_action_noise,
                        control_initial_value,
                    )  for _ in 1:3]
                treatment_params = [(
                        treatment_learning_rate,
                        treatment_action_noise,
                        treatment_initial_value,
                    ) for _ in 1:3]
                #Return vector
                return [control_params; treatment_params]   
            end

            parameter_names = [:learning_rate, :action_noise, :initial_value]

            model = create_model(
                action_model,
                custom_population_model(),
                data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = grouping_cols,
                parameter_names = parameter_names,
            )

            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "no grouping cols ($ad_type)" begin
            #Create model
            model = create_model(
                action_model,
                prior,
                data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = Symbol[],
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "multiple grouping cols ($ad_type)" begin

            #Create model
            model = create_model(
                action_model,
                prior,
                data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = grouping_cols,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)

        end

        @testset "missing actions ($ad_type)" begin

            #Create new dataframe where three actions = missing
            new_data = allowmissing(data, :actions)
            new_data[[2, 7, 12], :actions] .= missing

            #Create model
            model = create_model(
                action_model,
                prior,
                new_data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = grouping_cols,
                infer_missing_actions = true,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "multiple actions ($ad_type)" begin

            function multi_action(agent, input::Real)

                noise = agent.parameters[:noise]

                actiondist1 = Normal(input, noise)
                actiondist2 = Normal(input, noise)

                return (actiondist1, actiondist2)
            end
            #Create model
            new_model =
                ActionModel(multi_action, parameters = (; noise = Parameter(1.0, Real)))

            #Set prior
            new_prior = Dict(:noise => LogNormal(0.0, 1.0))

            #Create model
            model = create_model(
                new_model,
                new_prior,
                data,
                input_cols = input_cols,
                action_cols = [:actions, :actions_2],
                grouping_cols = grouping_cols,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "multiple actions, missing actions ($ad_type)" begin

            function multi_action(agent, input::Real)

                noise = agent.parameters[:noise]

                actiondist1 = Normal(input, noise)
                actiondist2 = Normal(input, noise)

                return (actiondist1, actiondist2)
            end
            #Create model
            new_model = ActionModel(multi_action, parameters = (;noise = Parameter(1.0, Real)))

            #Set prior
            new_prior = Dict(:noise => LogNormal(0.0, 1.0))

            #Create new dataframe where three actions = missing
            new_data = allowmissing(data, [:actions, :actions_2])
            new_data[[2, 12], :actions] .= missing
            new_data[[3], :actions_2] .= missing

            #Create model
            model = create_model(
                new_model,
                new_prior,
                new_data,
                input_cols = input_cols,
                action_cols = [:actions, :actions_2],
                grouping_cols = grouping_cols,
                infer_missing_actions = true,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "multiple inputs ($ad_type)" begin

            function multi_input(agent, input::Tuple{R,R}) where {R<:Real}

                noise = agent.parameters[:noise]

                input1, input2 = input

                actiondist = Normal(input1, noise)

                return actiondist
            end
            #Create model
            new_model = ActionModel(multi_input, parameters = (;noise = Parameter(1.0, Real)))

            new_prior = Dict(:noise => LogNormal(0.0, 1.0))

            #Create model
            model = create_model(
                new_model,
                new_prior,
                data,
                input_cols = [:inputs, :inputs_2],
                action_cols = action_cols,
                grouping_cols = grouping_cols,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "multiple inputs and multiple actions ($ad_type)" begin

            function multi_input_action(agent, input::Tuple{R,R}) where {R<:Real}

                noise = agent.parameters[:noise]

                input1, input2 = input

                actiondist1 = Normal(input1, noise)
                actiondist2 = Normal(input2, noise)

                return (actiondist1, actiondist2)
            end
            #Create action_model
            new_model = ActionModel(multi_input_action, parameters = (;noise = Parameter(1.0, Real)))

            new_prior = Dict(:noise => LogNormal(0.0, 1.0))

            #Create model
            model = create_model(
                new_model,
                new_prior,
                data,
                input_cols = [:inputs, :inputs_2],
                action_cols = [:actions, :actions_2],
                grouping_cols = grouping_cols,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end

        @testset "depend on previous action ($ad_type)" begin

            function dependent_action(agent::Agent, input::R) where {R<:Real}

                noise = agent.parameters[:noise]

                prev_action = agent.states[:action]

                if ismissing(prev_action)
                    prev_action = 0.0
                end

                actiondist = Normal(input + prev_action, noise)

                return actiondist
            end
            #Create model
            new_model = ActionModel(dependent_action, parameters = (;noise = Parameter(1.0, Real)))

            new_prior = Dict(:noise => LogNormal(0.0, 1.0))

            #Create model
            model = create_model(
                new_model,
                new_prior,
                data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = grouping_cols,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)

        end

        @testset "depend on previous action, multiple actions ($ad_type)" begin

            function dependent_multi_action(agent::Agent, input::R) where {R<:Real}

                noise = agent.parameters[:noise]

                prev_action = agent.states[:action]

                if ismissing(prev_action)
                    prev_action = 0.0
                end

                actiondist1 = Normal(input + prev_action, noise)
                actiondist2 = Normal(input - prev_action, noise)

                return (actiondist1, actiondist2)
            end
            #Create model
            new_model = ActionModel(dependent_multi_action, parameters = (;noise = Parameter(1.0, Real)))

            new_prior = Dict(:noise => LogNormal(0.0, 1.0))

            #Create model
            model = create_model(
                new_model,
                new_prior,
                data,
                input_cols = input_cols,
                action_cols = action_cols,
                grouping_cols = grouping_cols,
            )

            #Fit model
            chains = sample(model.model, sampler, n_iterations; sampling_kwargs...)
        end
    end
end
