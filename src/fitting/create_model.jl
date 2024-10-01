###########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A STATISTICAL MODEL ###
###########################################################################################################
function create_model(
    agent::Agent,
    statistical_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3} = Vector{String}(),
    verbose::Bool = true,
) where {T1<:Union{String,Symbol},T2<:Union{String,Symbol},T3<:Union{String,Symbol}}

    ## SETUP ##
    #Create a copy of the agent to avoid changing the original 
    agent_model = deepcopy(agent)

    #Turn off saving the history of states
    set_save_history!(agent_model, false)

    ## Make sure columns are vectors of symbols ##
    if !(input_cols isa Vector)
        input_cols = [input_cols]
    end
    if !(action_cols isa Vector)
        action_cols = [action_cols]
    end
    if !(grouping_cols isa Vector)
        grouping_cols = [grouping_cols]
    end
    input_cols = Symbol.(input_cols)
    action_cols = Symbol.(action_cols)
    grouping_cols = Symbol.(grouping_cols)

    #Run checks for the model specifications
    check_model(
        agent,
        statistical_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        verbose = verbose,
    )

    ## Extract data ##
    #One matrix per agent, for inputs and actions separately
    inputs =
        [Array(agent_data[:, input_cols]) for agent_data in groupby(data, grouping_cols)]
    actions =
        [Array(agent_data[:, action_cols]) for agent_data in groupby(data, grouping_cols)]

    #Create a full model combining the agent model and the statistical model
    return full_model(agent_model, statistical_model, inputs, actions)
end

###################################################################
### FUNCTION FOR DOING FULL AGENT AND STATISTICAL MODEL COMBINE ###
###################################################################
@model function full_model(
    agent::Agent,
    statistical_model::DynamicPPL.Model,
    inputs::Array{IA},
    actions::Array{AA},
    multiple_inputs::Bool = size(first(inputs), 2) > 1,
    multiple_actions::Bool = size(first(actions), 2) > 1,
) where {IAR<:Union{Real,Missing},AAR<:Union{Real,Missing},IA<:Array{IAR},AA<:Array{AAR}}

    #Check whether errors occur
    try

        #Generate the agent parameters from the statistical model
        @submodel statistical_model_return = statistical_model

        #Extract the agent parameters
        agents_parameters = statistical_model_return.agent_parameters

        #Initialize a vector for storing the states of the agents
        parameters_per_agent = Vector{Dict}(undef, length(agents_parameters))

        ## For each agent ##
        for (agent_idx, agent_parameters) in enumerate(agents_parameters)

            #Set the agent parameters
            set_parameters!(agent, agent_parameters)
            reset!(agent)

            ## Construct input iterator ##
            #If there is only one input
            if !multiple_inputs
                #Iterate over inputs one at a time
                input_iterator = enumerate(inputs[agent_idx])
            else
                #Iterate over rows of inputs
                input_iterator = enumerate(Tuple.(eachrow(inputs[agent_idx])))
            end

            #Go through each timestep 
            for (timestep, input) in input_iterator

                ## Sample actions ##

                #Get the action probability distributions from the action model
                action_distribution = agent.action_model(agent, input)

                #If there is only one action
                if !multiple_actions

                    #Sample the action from the probability distribution
                    @inbounds actions[agent_idx][timestep] ~ action_distribution

                    #Save the action to the agent in case it needs it in the future
                    @inbounds update_states!(
                        agent,
                        "action",
                        ad_val.(actions[agent_idx][timestep]),
                    )

                    #If there are multiple actions
                else
                    #Go through each separate action
                    for (action_idx, single_distribution) in enumerate(action_distribution)

                        #Sample the action from the probability distribution
                        @inbounds actions[agent_idx][timestep, action_idx] ~
                            single_distribution
                    end

                    #Add the actions to the agent in case it needs it in the future
                    @inbounds update_states!(
                        agent,
                        "action",
                        ad_val.(actions[agent_idx][timestep, :]),
                    )
                end
            end
            #Save the parameters of the agent
            parameters_per_agent[agent_idx] = get_parameters(agent)
        end

        #Return agents' parameters and tracked states
        return GeneratedQuantitites(
            parameters_per_agent,
            statistical_model_return.statistical_values,
        )

        #If an error occurs
    catch error
        #If it is of the custom errortype RejectParameters
        if error isa RejectParameters
            #Make Turing reject the sample
            Turing.@addlogprob!(-Inf)
        else
            #Otherwise, just throw the error
            rethrow(error)
        end
    end
end
