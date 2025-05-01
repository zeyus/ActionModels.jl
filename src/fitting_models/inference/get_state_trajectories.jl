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
    all_session_parameters = get_session_parameters!(modelfit, prior_or_posterior).value
    #Extract the agent
    agent = deepcopy(model.args.agent_model)
    set_save_history!(agent, true)
    #Extract observations
    observations_per_session = model.args.observations_per_session

    #Extract dimension labels
    session_ids, estimated_parameter_names, sample_idxs, chain_idxs = all_session_parameters.axes

    #Make parameter names into a vector
    estimated_parameter_names = collect(estimated_parameter_names)
    session_ids = collect(session_ids)


    ### Checks ###
    #If any of the target states are not in agent_states, throw an error
    agent_states = collect(keys(get_states(agent)))
    for target_state in target_states
        if !(target_state in agent_states)
            error("Target state $target_state not found in agent states.")
        end
    end

    ### Extract States ###
    #Loop through sessions
    state_trajectories = [
        begin

            #Extract the observations for the current session
            session_observations = observations_per_session[session_idx]

            #Create empty AxisArray for the session
            session_trajectories = AxisArray(
                Array{Union{Missing,Real}}(
                    undef,
                    length(session_observations) + 1,
                    length(target_states),
                    length(sample_idxs),
                    length(chain_idxs),
                ),
                Axis{:timestep}(0:length(session_observations)),
                Axis{:state}(Symbol.(target_states)),
                Axis{:sample}(1:sample_idxs[end]),
                Axis{:chain}(1:chain_idxs[end]),
            )

            #For each sample and each chain
            for (sample_idx, chain_idx) in
                Iterators.product(1:length(sample_idxs), 1:length(chain_idxs))

                parameter_sample = Tuple(session_parameters[:, sample_idx, chain_idx])

                #Set parameters in agent
                set_parameters!(agent, estimated_parameter_names, parameter_sample)
                reset!(agent)

                #Go through each observation
                give_observations!(agent, session_observations)

                #Extract histories
                state_histories =
                    hcat([get_history(agent, state) for state in target_states]...)

                #Store them in the session_trajectories
                session_trajectories[:, :, sample_idx, chain_idx] .= state_histories
            end

            #Return the session_trajectories
            session_trajectories

        end for (session_idx, (session_observations, session_parameters)) in
        enumerate(zip(observations_per_session, eachslice(all_session_parameters, dims = 1)))
    ]

    #Return StateTrajectories struct
    return StateTrajectories(
        target_states,
        session_ids,
        state_trajectories,
    )
end