##################################
####### NO MISSING ACTIONS #######
##################################
function create_session_model(
    infer_missing_actions::Nothing,         #No missing actions
    multiple_actions::Val,                  #Single or multiple actions
    check_parameter_rejections::Val{false}, #No parameter rejections
    actions::Vector{Vector{A}},
) where {A}

    #Create flattened actions
    flattened_actions = evert(collect(Iterators.flatten(actions)))

    #function for sampling a single action idx
    @model function single_action_idx(
        actions::Vector{R},
        distributions::Vector{D},
    ) where {R<:Real,D<:Distribution}
        actions ~ arraydist(distributions)
    end

    #Create session model function with flattened actions included
    return @model function my_session_model(
        agent::Agent,
        parameter_names::Vector{Symbol},
        session_ids::Vector{String},
        parameters_per_session::T, #No way to type for an iterator
        inputs_per_session::Vector{Vector{II}},
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
                set_parameters!(agent, parameter_names, session_parameters)
                reset!(agent)
                [
                    begin
                        #Get the action probability (either a distribution, or a tuple of distributions) 
                        action_distribution = agent.action_model(agent, input)
                        #Save the action
                        update_states!(agent, :action, action)

                        #Return the action probability distribution
                        action_distribution

                    end for (input, action) in zip(session_inputs, session_actions)
                ]
            end for (session_parameters, session_inputs, session_actions) in
            zip(parameters_per_session, inputs_per_session, actions_per_session)
        ]

        ## Reshape into a tuple of vectors with distributions ## 
        flattened_distributions = evert(collect(Iterators.flatten(action_distributions)))

        ## Sample the actions from the probability distributions ##
        for (actions, distributions) in zip(flattened_actions, flattened_distributions)
            a ~ to_submodel(single_action_idx(actions, distributions), false)
        end
    end
end



#####################################
####### INFER MISSING ACTIONS #######
#####################################
function create_session_model(
    infer_missing_actions::InferMissingActions,         #Infer missing actions
    multiple_actions::Val,                              #One or multiple actions
    check_parameter_rejections::Val{false},             #No parameter rejections
    actions::Vector{Vector{A}},
) where {A}

    #Create prefixes for the submodels
    prefixes = [
        [Symbol(string("session_", i, ".timestep_", j)) for j = 1:length(actions[i])]
        for i = 1:length(actions)
    ]

    #Create session model function
    return @model function my_session_model(
        agent::Agent,
        parameter_names::Vector{Symbol},
        session_ids::Vector{String},
        parameters_per_session::T, #No way to type for an iterator
        inputs_per_session::Vector{Vector{II}},
        actions_per_session::Vector{Vector{AA}};
        prefixes_per_session::Vector{Vector{Symbol}} = prefixes,
    ) where {
        I<:Any,
        II<:Union{I,Tuple{Vararg{I}}},
        A<:Union{<:Real,Missing},
        AA<:Union{A,<:Tuple{Vararg{Union{Missing,A}}}},
        T<:Any,
    }

        #For each session
        for (
            session_parameters,
            session_inputs,
            session_actions,
            session_prefixes,
            session_id,
        ) in zip(
            parameters_per_session,
            inputs_per_session,
            actions_per_session,
            prefixes_per_session,
            session_ids,
        )

            i ~ to_submodel(
                prefix(
                    sample_single_session(
                        agent,
                        parameter_names,
                        session_parameters,
                        session_inputs,
                        session_actions,
                        session_prefixes,
                    ),
                    session_id,
                ),
                false,
            )
        end
    end
end

@model function sample_single_session(
    agent::Agent,
    parameter_names::Vector{Symbol},
    session_parameters::T,
    session_inputs::Vector{II},
    session_actions::Vector{AA},
    session_prefixes::Vector{Symbol},
) where {
    I<:Any,
    II<:Union{I,Tuple{Vararg{I}}},
    A<:Union{Real,Missing},
    AA<:Union{A,<:Tuple{Vararg{Union{Missing,A}}}},
    T<:Tuple,
}
    #Prepare the agent
    set_parameters!(agent, parameter_names, session_parameters)
    reset!(agent)

    #For each timestep
    for (input, action, timestep_prefix) in
        zip(session_inputs, session_actions, session_prefixes)

        i ~ to_submodel(
            prefix(sample_single_timestep(agent, input, action), timestep_prefix),
            false,
        )
    end
end

#Turing submodel for sampling a simple timestep, with a single action
@model function sample_single_timestep(
    agent::Agent,
    input::I,
    action::A,
) where {I<:Any,A<:Union{Real,Missing}}
    #Give input and sample action
    action ~ agent.action_model(agent, input)

    #Store the agent's action in the agent
    update_states!(agent, :action, action)
end

#Turing submodel for sampling a simple timestep, with multiple actions
@model function sample_single_timestep(
    agent::Agent,
    input::I,
    actions::AA,
) where {I<:Any,A<:Union{Real,Missing},AA<:Union{A,<:Tuple{Vararg{Union{Missing,A}}}}}

    #Get the tuple of action distributions from the action model
    action_distributions = agent.action_model(agent, input)

    action = Tuple(
        i ~ to_submodel(
            prefix(sample_subaction(action, action_distribution), "index_$action_idx"),
            false,
        ) for (action_idx, (action, action_distribution)) in
        enumerate(zip(actions, action_distributions))
    )

    #Store the agent's action in the agent
    update_states!(agent, :action, action)
end

#Turing subsubmodel for sampling one of the actions in the timesteo
@model function sample_subaction(
    action::A,
    action_distribution::D,
) where {A<:Union{<:Real,Missing},D<:Distribution}
    action ~ action_distribution

    return action
end



####################################
####### SKIP MISSING ACTIONS #######
####################################
function create_session_model(
    infer_missing_actions::SkipMissingActions,  #Skip missing actions
    multiple_actions::Val,                      #Single or multiple actions
    check_parameter_rejections::Val{false},     #No parameter rejections
    actions::Vector{Vector{A}},
) where {A}

    throw(ArgumentError("Skipping missing actions is not yet supported"))

end






####################################
####### PARAMETER REJECTIONS #######
####################################
function create_session_model(
    infer_missing_actions::Union{Nothing,SkipMissingActions,InferMissingActions},  #Any missing actions
    multiple_actions::Val,                                                           #Single or multiple actions
    check_parameter_rejections::Val{true},                                           #No parameter rejections
    actions::Vector{Vector{A}},
) where {A}

    throw(ArgumentError("Checking for parameter rejections is not yet supported"))

end