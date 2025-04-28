##################################
####### NO MISSING ACTIONS #######
##################################
function create_session_model(
    infer_missing_actions::NoMissingActions,    #No missing actions
    multiple_actions::AbstractMultipleActions,  #Single or multiple actions
    check_parameter_rejections::Val{false},     #No parameter rejections
    actions::Vector{Vector{A}},
) where {A}

    #Create flattened actions
    flattened_actions = evert(collect(Iterators.flatten(actions)))

    #Submodel for sampling all actions of a single action idx as an arraydist
    @model function sample_actions_one_idx(
        actions::Vector{R},
        distributions::Vector{D},
    ) where {R<:Real,D<:Distribution}
        actions ~ arraydist(distributions)
    end

    #Create session model function with flattened actions included
    return @model function my_session_model(
        agent::Agent,
        estimated_parameters::Vector{Symbol},
        session_ids::Vector{String},
        parameters_per_session::T, #No way to type for an iterator
        observations_per_session::Vector{Vector{II}},
        actions_per_session::Vector{Vector{AA}};
        flattened_actions::FA = flattened_actions,
    ) where {
        I<:Any,
        II<:Union{I,Tuple{Vararg{I}}},
        A<:Real,
        AA<:Union{A,Tuple{Vararg{A}}},
        FA<:Tuple,
        T<:Any,
    }

        ## Run forwards to get the action distributions ##
        action_distributions = [
            begin
                #Set the agent parameters
                set_parameters!(agent, estimated_parameters, session_parameters)
                reset!(agent)
                [
                    begin
                        #Get the action probability (either a distribution, or a tuple of distributions) 
                        action_distribution = agent.action_model(agent, observation...)
                        #Save the action
                        update_states!(agent, :action, action)

                        #Return the action probability distribution
                        action_distribution

                    end for
                    (observation, action) in zip(session_observations, session_actions)
                ]
            end for (session_parameters, session_observations, session_actions) in
            zip(parameters_per_session, observations_per_session, actions_per_session)
        ]

        ## Reshape into a tuple of vectors with distributions ## 
        flattened_distributions = evert(collect(Iterators.flatten(action_distributions)))

        ## Sample the actions from the probability distributions ##
        for (actions, distributions) in zip(flattened_actions, flattened_distributions)
            a ~ to_submodel(sample_actions_one_idx(actions, distributions), false)
        end
    end
end

####################################
####### WITH MISSING ACTIONS #######
####################################
function create_session_model(
    infer_missing_actions::AbstractMissingActions,
    multiple_actions::AbstractMultipleActions,
    check_parameter_rejections::Val{false},
    actions::Vector{Vector{A}},
) where {A}

    ## Submodel for sampling all actions of a single action idx as an arraydist ##
    @model function sample_actions_one_idx(
        actions::Vector{R},
        distributions::Vector{D},
    ) where {R<:Any,D<:Any}
        # ) where {R<:Real,D<:Distribution} TODO: once things have been made typestable we can clean this
        actions ~ arraydist(distributions)
    end

    ## Create timestep prefixes ##
    timestep_prefixes = [
        [Symbol(string("timestep_", j)) for j = 1:length(actions[i])] for
        i = 1:length(actions)
    ]

    ## Find timesteps with missing actions ##
    missing_action_markers = [
        Vector{AbstractMissingActions}(undef, length(session_actions)) for
        session_actions in actions
    ]
    #Go through each session
    for (session_idx, session_actions) in enumerate(actions)
        for (i, action) in enumerate(session_actions)

            #If it is missing, set the marker to the appropriate type
            if action isa Missing
                missing_action_markers[session_idx][i] = infer_missing_actions
            elseif action isa Tuple && any(x -> x isa Missing, action)
                missing_action_markers[session_idx][i] = infer_missing_actions
            else
                missing_action_markers[session_idx][i] = NoMissingActions()
            end
        end
    end

    ## Create flattened actions ##
    flattened_missing_action_markers = collect(Iterators.flatten(missing_action_markers))
    flattened_actions = collect(Iterators.flatten(actions))
    #Find the missing action markers
    flattened_actions = [
        marker isa NoMissingActions ? action : nothing for (action, marker) in zip(
            flattened_actions, flattened_missing_action_markers
        )]
    #Filter out the missing actions and evert
    flattened_actions = filter(x -> !isnothing(x), flattened_actions)
    #Ensure type stability (TODO: once types are pre-inferred we can make this smarter)
    flattened_actions = [action for action in flattened_actions]
    #Make into a tuple of vectors
    flattened_actions = evert(flattened_actions)

    #Create session model function
    return @model function session_model(
        agent::Agent,
        estimated_parameters::Vector{Symbol},
        session_ids::Vector{String},
        parameters_per_session::T, #No way to type for an iterator
        observations_per_session::Vector{Vector{II}},
        actions_per_session::Vector{Vector{AA}};
        timestep_prefixes_per_session::Vector{Vector{Symbol}} = timestep_prefixes,
        missing_action_markers_per_session::Vector{Vector{AbstractMissingActions}} = missing_action_markers,
        multiple_actions::AbstractMultipleActions = multiple_actions,
        flattened_actions::FA = flattened_actions,
    ) where {
        I<:Any,
        II<:Union{I,Tuple{Vararg{I}}},
        A<:Union{<:Real,Missing},
        AA<:Union{A,<:Tuple{Vararg{Union{Missing,A}}}},
        T<:Any,
        FA<:Tuple,
    }
        ## Run forwards to get the action distributions ##
        action_distributions = [
            i ~ to_submodel(
                prefix(
                    single_session_model(
                        agent,
                        estimated_parameters,
                        session_parameters,
                        session_observations,
                        session_actions,
                        session_timestep_prefixes,
                        session_missing_action_markers,
                        multiple_actions,
                    ),
                    session_id,
                ),
                false,
            ) for (
                session_parameters,
                session_observations,
                session_actions,
                session_timestep_prefixes,
                session_missing_action_markers,
                session_id,
            ) in zip(
                parameters_per_session,
                observations_per_session,
                actions_per_session,
                timestep_prefixes_per_session,
                missing_action_markers_per_session,
                session_ids,
            )
        ]

        #Remove missing action distributions
        flattened_distributions = filter(x -> !isnothing(x), collect(Iterators.flatten(action_distributions)))

        #Ensure type stability (TODO: once types are pre-inferred we can make this smarter)
        flattened_distributions = [distribution for distribution in flattened_distributions]
        #Make distributions into a tuple of vectors
        flattened_distributions = evert(flattened_distributions)

        ## Sample the actions from the probability distributions ##
        for (actions, distributions) in zip(flattened_actions, flattened_distributions)
            a ~ to_submodel(sample_actions_one_idx(actions, distributions), false)
        end
    end
end


#######################################
####### SINGLE SESSION SUBMODEL #######
#######################################
@model function single_session_model(
    agent::Agent,
    estimated_parameters::Vector{Symbol},
    session_parameters::T,
    session_observations::Vector{II},
    session_actions::Vector{AA},
    session_timestep_prefixes::Vector{Symbol},
    session_missing_action_markers::Vector{AbstractMissingActions},
    multiple_actions::AbstractMultipleActions,
) where {
    I<:Any,
    II<:Union{I,Tuple{Vararg{I}}},
    A<:Union{Real,Missing},
    AA<:Union{A,<:Tuple{Vararg{Union{Missing,A}}}},
    T<:Tuple,
}
    #Prepare the agent
    set_parameters!(agent, estimated_parameters, session_parameters)
    reset!(agent)

    session_action_distributions = [
        i ~ to_submodel(
            prefix(
                sample_single_timestep(
                    agent,
                    observation,
                    action,
                    missing_action_marker,
                    multiple_actions,
                ),
                timestep_prefix,
            ),
            false,
        ) for (observation, action, timestep_prefix, missing_action_marker) in zip(
            session_observations,
            session_actions,
            session_timestep_prefixes,
            session_missing_action_markers,
        )
    ]
end

#########################################
####### SINGLE TIMESTEP SUBMODELS #######
#########################################

@model function sample_single_timestep(
    agent::Agent,
    observation::I,
    action::A,
    missing_actions::NoMissingActions,      #No missing actions
    multiple_actions::T,                    #Single or multiple actions
) where {I<:Any,A<:Union{Real,Tuple{Vararg{Real}}},T<:AbstractMultipleActions}
    #Give observation and get action distribution
    action_distribution = agent.action_model(agent, observation...)

    #Store the agent's action in the agent
    update_states!(agent, :action, action)

    #Return action distribution
    return action_distribution
end

@model function sample_single_timestep(
    agent::Agent,
    observation::I,
    action::A,
    missing_actions::SkipMissingActions,    #Skip missing actions
    multiple_actions::T,                    #Single or multiple actions  
) where {
    I<:Any,
    A<:Union{Missing,Tuple{Vararg{Union{Missing,Real}}}},
    T<:AbstractMultipleActions,
}
    #Give observation and get action distribution
    action_distribution = agent.action_model(agent, observation...)

    #Store the agent's action in the agent
    update_states!(agent, :action, action)

    #Return nothing
    return nothing
end


@model function sample_single_timestep(
    agent::Agent,
    observation::I,
    action::A,
    missing_actions::InferMissingActions,   #Infer missing actions
    multiple_actions::SingleAction,         #Single action
) where {I<:Any,A<:Missing}
    #Give observation and sample the missing action
    action ~ agent.action_model(agent, observation...)

    #Store the agent's sampled action in the agent
    update_states!(agent, :action, action)

    return nothing
end


@model function sample_single_timestep(
    agent::Agent,
    observation::I,
    actions::A,
    missing_actions::InferMissingActions,   #Infer missing actions
    multiple_actions::MultipleActions,      #Multiple actions
) where {I<:Any,A<:Tuple{Vararg{Union{Missing,Real}}}}

    #Get the tuple of action distributions from the action model
    action_distributions = agent.action_model(agent, observation...)

    #Sample the actions separately from the probability distributions
    action = Tuple(
        i ~ to_submodel(
            prefix(sample_subaction(action, action_distribution), "subaction_$action_idx"),
            false,
        ) for (action_idx, (action, action_distribution)) in
        enumerate(zip(actions, action_distributions))
    )

    #Store the agent's action in the agent
    update_states!(agent, :action, action)

    return nothing
end

#Turing subsubmodel for sampling one of the actions in the timestep
@model function sample_subaction(
    action::A,
    action_distribution::D,
) where {A<:Union{<:Real,Missing},D<:Distribution}
    action ~ action_distribution

    return action
end



####################################
####### PARAMETER REJECTIONS #######
####################################
function create_session_model(
    infer_missing_actions::AbstractMissingActions,  
    multiple_actions::AbstractMultipleActions,     
    check_parameter_rejections::Val{true},               #No parameter rejections
    actions::Vector{Vector{A}},
) where {A}

    #Get normal session model
    session_submodel = create_session_model(
        infer_missing_actions,
        multiple_actions,
        Val{false}(),
        actions,
    )

    return @model function session_model(
        agent::Agent,
        estimated_parameters::Vector{Symbol},
        session_ids::Vector{String},
        parameters_per_session::T, #No way to type for an iterator
        observations_per_session::Vector{Vector{II}},
        actions_per_session::Vector{Vector{AA}};
    ) where {
        I<:Any,
        II<:Union{I,Tuple{Vararg{I}}},
        A<:Real,
        AA<:Union{A,Tuple{Vararg{A}}},
        T<:Any,
    }

        try 

        #Run the normal session model
        i ~ to_submodel(session_submodel(
            agent,
            estimated_parameters,
            session_ids,
            parameters_per_session,
            observations_per_session,
            actions_per_session,
        ), false)

        catch e
            #If there is a parameter rejection, reject the sample
            if isa(e, RejectParameters)
                Turing.@addlogprob! -Inf
                return nothing
            else
                rethrow(e)
            end
        end
    end
end