function get_state_trajectories!(
    modelfit::ModelFit,
    target_states::Union{Symbol,Vector{Symbol}},
    prior_or_posterior::Symbol = :posterior;
)

    ### Setup ###
    #Make target states into a vector
    if target_states isa Symbol
        target_states = [target_states]
    end

    #Extract the model
    model = modelfit.model
    #Extract the appropirate session parameters
    all_session_parameters = get_session_parameters!(modelfit, prior_or_posterior)
    #create an agent
    agent = init_agent(model.args.action_model, save_history = target_states)

    #Extract observations
    observations_per_session = model.args.observations_per_session

    #Extract dimension labels
    session_ids = all_session_parameters.session_ids
    estimated_parameter_names = all_session_parameters.estimated_parameter_names
    n_samples = all_session_parameters.n_samples
    n_chains = all_session_parameters.n_chains

    #Extract state types
    action_model = model.args.action_model
    state_types =
        merge(get_state_types(action_model), get_state_types(action_model.submodel))

    ### Checks ###
    #If any of the target states are not in agent_states, throw an error
    agent_states = collect(keys(get_states(agent)))
    for target_state in target_states
        if !(target_state in agent_states)
            error("Target state $target_state not found in agent states.")
        end
    end

    ## Prepare datacontainer for populating ##
    state_trajectories = NamedTuple(
        state_name => NamedTuple(
            Symbol(session_id) => AxisArray(
                Array{state_types[state_name]}(
                    undef,
                    n_samples,
                    n_chains,
                    length(session_observations) + 1,
                ),
                Axis{:sample}(1:n_samples),
                Axis{:chain}(1:n_chains),
                Axis{:timestep}(0:length(session_observations)),
            ) for (session_id, session_observations) in
            zip(session_ids, observations_per_session)
        ) for state_name in target_states
    )

    ### Extract States ###
    #Loop through sessions
    @progress for session_idx = 1:length(session_ids)

        #Extract session observations
        session_observations = observations_per_session[session_idx]

        #Loop through samples
        for (sample_idx, chain_idx) in Iterators.product(1:n_samples, 1:n_chains)

            #Extract the parameter sample
            parameter_sample = map(parameter_name -> all_session_parameters.value[parameter_name][session_idx][sample_idx, chain_idx],
                estimated_parameter_names,
            )

            #Set parameters in agent
            set_parameters!(agent, estimated_parameter_names, parameter_sample)
            reset!(agent)

            #Go through each observation
            simulate!(agent, session_observations)

            #Loop through target states
            for state in target_states

                #Extract the state history
                state_history = get_history(agent, state)

                #Store it in the session_trajectories
                state_trajectories[state][session_idx][sample_idx, chain_idx, :] .= state_history
            end
        end
    end

    #Return StateTrajectories struct
    return StateTrajectories(state_trajectories, target_states, session_ids, state_types, n_samples, n_chains)
end
