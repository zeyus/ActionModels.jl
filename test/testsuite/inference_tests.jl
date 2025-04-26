using Test

using ActionModels
using DataFrames
using Turing: AutoForwardDiff, AutoReverseDiff, AutoMooncake

@testset "inference tests" begin

    ### SETUP ###
    #Generate dataset
    data = DataFrame(
        observations = repeat([1.0, 1, 1, 2, 2, 2], 6),
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

    #Add a second set of actions and observations
    data.actions_2 = data.actions
    data.observations_2 = data.observations

    #Define observation and action cols
    observation_cols = [:observations]
    action_cols = [:actions]
    grouping_cols = [:id, :treatment]

    #Create model
    action_model = ActionModel(ContinuousRescorlaWagnerGaussian())

    #Set prior
    prior = Dict(
        :learning_rate => LogitNormal(),
        :action_noise => LogNormal(),
        :initial_value => Normal(),
    )

    #Inference parameters
    n_samples = 50
    n_chains = 2

    #Go through each supported AD type
    for AD in
        ["AutoForwardDiff", "AutoReverseDiff", "AutoReverseDiff(true)", "AutoMooncake"]

        #Select appropriate AD backend
        if AD == "AutoForwardDiff"
            ad_type = AutoForwardDiff()
        elseif AD == "AutoReverseDiff"
            ad_type = AutoReverseDiff()
        elseif AD == "AutoReverseDiff(true)"
            ad_type = AutoReverseDiff(; compile = true)
        elseif AD == "AutoMooncake"
            ad_type = AutoMooncake(; config = nothing)
        end

        @testset "API tests $(AD)" begin
            
            #Create model
            model = create_model(
                action_model,
                prior,
                data,
                observation_cols = observation_cols,
                action_cols = action_cols,
                grouping_cols = grouping_cols,
            )

            @testset "sampling and extracting results $(AD)" begin
                #Posterior
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
                posterior_parameters = get_session_parameters!(model, :posterior)
                summarize(posterior_parameters)
                summarize(posterior_parameters, mean)
                posterior_trajectories =
                    get_state_trajectories!(model, [:observation, :value], :posterior)
                summarize(posterior_trajectories)
                summarize(posterior_trajectories, mean)

                #Prior
                prior_chains =
                    sample_prior!(model, n_samples = n_samples, n_chains = n_chains)
                prior_parameters = get_session_parameters!(model, :prior)
                summarize(prior_parameters)
                prior_trajectories =
                    get_state_trajectories!(model, [:observation, :value], :prior)
                summarize(prior_trajectories)
            end

            @testset "plotting results $(AD)" begin
                ### Test plotting functions ###
                using StatsPlots

                #Default Turing plots
                plot(model.posterior.chains)
                plot(model.prior.chains)

                # plot(posterior_parameters)
                # plot(posterior_parameters, session = "id:Hans.treatment:control")
                # plot(posterior_trajectories)
                # plot(posterior_trajectories, session = "id:Hans.treatment:control")

                # plot(prior_parameters)
                # plot(prior_parameters, session = "id:Hans.treatment:control")
                # plot(prior_trajectories)
                # plot(prior_trajectories, session = "id:Hans.treatment:control")
            end

            @testset "sample_posterior! variations $(AD)" begin

                @testset "different init_params $(AD)" begin
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = nothing,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = :MLE,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = :MAP,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = :sample_prior,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                end

                @testset "multi-core sampling $(AD)" begin
                    # using Distributed

                    # addprocs(2)

                    # @everywhere using ActionModels
                    # @everywhere model = $model

                    # posterior_chains = sample_posterior!(model, MCMCDistributed(), resample = true, n_samples = n_samples, n_chains = n_chains)

                    # rmprocs(workers())
                end

                @testset "multi-thread sampling $(AD)" begin
                    #TODO:
                end

                @testset "save/resume THIS ERRORS! $(AD)" begin
                    # posterior_chains = sample_posterior!(model, resample = true, save_resume = SampleSaveResume(path = mktempdir()), n_samples = n_samples, n_chains = n_chains)
                end

                @testset "multi-core save/resume $(AD)" begin
                    # using Distributed

                    # addprocs(2)

                    # @everywhere using ActionModels
                    # @everywhere model = $model

                    # posterior_chains = sample_posterior!(model, MCMCDistributed(), save_resume = SampleSaveResume(path = mktempdir()), resample = true, n_samples = n_samples, n_chains = n_chains)

                    # rmprocs(workers())
                end


                @testset "multi-thread save/resume $(AD)" begin
                    #TODO:
                end
            end
        end

        @testset "population model tests $(AD)" begin

            ### TESTING MODEL TYPES ###
            @testset "single session ($AD)" begin
                #Extract observations and actions from data
                observations = data[!, :observations]
                actions = data[!, :actions]

                #Create model
                model = create_model(action_model, prior, observations, actions)

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "simple statistical model ($AD)" begin
                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    grouping_cols = [:id],
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "custom population model ($AD)" begin

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
                    control_params = [
                        (
                            control_learning_rate,
                            control_action_noise,
                            control_initial_value,
                        ) for _ = 1:3
                    ]
                    treatment_params = [
                        (
                            treatment_learning_rate,
                            treatment_action_noise,
                            treatment_initial_value,
                        ) for _ = 1:3
                    ]
                    #Return vector
                    return [control_params; treatment_params]
                end

                parameter_names = [:learning_rate, :action_noise, :initial_value]

                model = create_model(
                    action_model,
                    custom_population_model(),
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    grouping_cols = grouping_cols,
                    parameter_names = parameter_names,
                )

                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
        end


        @testset "different observation and action types ($AD)" begin

            @testset "no grouping cols ($AD)" begin
                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple grouping cols ($AD)" begin

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    grouping_cols = grouping_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "missing actions ($AD)" begin

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, :actions)
                new_data[[2, 7, 12], :actions] .= missing

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    grouping_cols = grouping_cols,
                    infer_missing_actions = true,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple actions ($AD)" begin

                function multi_action(agent, observation::Float64)

                    noise = agent.parameters[:noise]

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0, Real)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                #Set prior
                new_prior = Dict(:noise => LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    grouping_cols = grouping_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple actions, missing actions ($AD)" begin

                function multi_action(agent, observation::Float64)

                    noise = agent.parameters[:noise]

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

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
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    grouping_cols = grouping_cols,
                    infer_missing_actions = true,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple observations ($AD)" begin

                function multi_observation(
                    agent,
                    observation1::Float64,
                    observation2::Float64,
                )

                    noise = agent.parameters[:noise]

                    actiondist = Normal(observation1, noise)

                    return actiondist
                end
                #Create model
                new_model = ActionModel(
                    multi_observation,
                    parameters = (; noise = Parameter(1.0, Float64)),
                    observations = (;
                        observation_1 = Observation(Float64),
                        observation_2 = Observation(Float64),
                    ),
                    actions = (; action_1 = Action(Normal)),
                )

                new_prior = Dict(:noise => LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = [:observations, :observations_2],
                    action_cols = action_cols,
                    grouping_cols = grouping_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple observations and multiple actions ($AD)" begin

                function multi_observation_action(
                    agent,
                    observation1::Float64,
                    observation2::Float64,
                )

                    noise = agent.parameters[:noise]

                    actiondist1 = Normal(observation1, noise)
                    actiondist2 = Normal(observation2, noise)

                    return (actiondist1, actiondist2)
                end
                #Create action_model
                new_model = ActionModel(
                    multi_observation_action,
                    parameters = (; noise = Parameter(1.0, Float64)),
                    observations = (;
                        observation_1 = Observation(Float64),
                        observation_2 = Observation(Float64),
                    ),
                    actions = (; action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                new_prior = Dict(:noise => LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = [:observations, :observations_2],
                    action_cols = [:actions, :actions_2],
                    grouping_cols = grouping_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "single session with multiple observations and actions ($AD)" begin

                function multi_observation_action(
                    agent,
                    observation1::Float64,
                    observation2::Float64,
                )

                    noise = agent.parameters[:noise]

                    actiondist1 = Normal(observation1, noise)
                    actiondist2 = Normal(observation2, noise)

                    return (actiondist1, actiondist2)
                end

                #Create action_model
                new_model = ActionModel(
                    multi_observation_action,
                    parameters = (; noise = Parameter(1.0, Float64)),
                    observations = (;
                        observation_1 = Observation(Float64),
                        observation_2 = Observation(Float64),
                    ),
                    actions = (; action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                new_prior = Dict(:noise => LogNormal(0.0, 1.0))

                #Extract observations and actions from data
                observations = Tuple.(eachrow(data[!, [:observations, :observations_2]]))
                actions = Tuple.(eachrow(data[!, [:actions, :actions_2]]))

                #Create model
                model = create_model(new_model, new_prior, observations, actions)

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "depend on previous action ($AD)" begin

                function dependent_action(agent::Agent, observation::Float64)

                    noise = agent.parameters[:noise]

                    prev_action = agent.states[:action]

                    if ismissing(prev_action)
                        prev_action = 0.0
                    end

                    actiondist = Normal(observation + prev_action, noise)

                    return actiondist
                end
                #Create model
                new_model = ActionModel(
                    dependent_action,
                    parameters = (; noise = Parameter(1.0, Real)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action_1 = Action(Normal)),
                )

                new_prior = Dict(:noise => LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    grouping_cols = grouping_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "depend on previous action, multiple actions ($AD)" begin

                function dependent_multi_action(agent::Agent, observation::Float64)

                    noise = agent.parameters[:noise]

                    prev_action = agent.states[:action]

                    if ismissing(prev_action)
                        prev_action = (0.0, 0.0)
                    end

                    actiondist1 = Normal(observation + prev_action[1], noise)
                    actiondist2 = Normal(observation - prev_action[2], noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    dependent_multi_action,
                    parameters = (; noise = Parameter(1.0, Real)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                new_prior = Dict(:noise => LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    grouping_cols = grouping_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
        end
    end
end