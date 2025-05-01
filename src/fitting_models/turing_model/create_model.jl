##########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A POPULATION MODEL ###
##########################################################################################################
function create_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    observation_cols::Union{
        NamedTuple{observation_names_cols,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names_cols,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    grouping_cols::Union{Vector{Symbol},Symbol} = Vector{Symbol}(),
    parameters_to_estimate::Vector{Symbol},
    infer_missing_actions::Bool = false,
    check_parameter_rejections::Bool = false,
    population_model_type::AbstractPopulationModel = CustomPopulationModel(),
    verbose::Bool = true,
) where {observation_names_cols,action_names_cols}

    ### ARGUMENT SETUP & CHECKS ###
    #Check that observation cols and action cols are the same length as the observations and actions in the action model
    if length(observation_cols) != length(action_model.observations)
        throw(
            ArgumentError(
                "The number of observation columns does not match the number of observations in the action model",
            ),
        )
    end
    if length(action_cols) != length(action_model.actions)
        throw(
            ArgumentError(
                "The number of action columns does not match the number of actions in the action model",
            ),
        )
    end

    ## Change columns to the correct format ##
    #Make single action and observation columns into vectors
    if observation_cols isa Symbol
        observation_cols = [observation_cols]
    end
    if action_cols isa Symbol
        action_cols = [action_cols]
    end
    #Make sure that observation_cols and action_cols are named tuples
    if observation_cols isa Vector
        observation_cols = NamedTuple{keys(action_model.observations)}(observation_cols)
        if verbose && length(observation_cols) > 1
            @warn "Mappings from action model observations to observation columns not provided. Using the order from the action model: $(observation_cols)"
        end
    end
    if action_cols isa Vector
        action_cols = NamedTuple{keys(action_model.actions)}(action_cols)
        if verbose && length(action_cols) > 1
            @warn "Mappings from action model actions to action columns not provided. Using the order from the action model: $(action_cols)"
        end
    end
    #Order observation and action columns to match the action model
    observation_cols = NamedTuple(
        observation_name => observation_cols[observation_name] for
        observation_name in keys(action_model.observations)
    )
    action_cols = NamedTuple(
        action_name => action_cols[action_name] for
        action_name in keys(action_model.actions)
    )

    #Grouping columns are a vector of symbols
    if !(grouping_cols isa Vector)
        grouping_cols = [grouping_cols]
    end

    ## Check whether to skip or infer missing data ##
    if !infer_missing_actions
        #If there are no missing actions
        if !any(ismissing, Matrix(data[!, collect(action_cols)]))
            #Remove any potential Missing type
            disallowmissing!(data, collect(action_cols))
            infer_missing_actions = NoMissingActions()
        else
            if verbose
                @warn """
                      There are missing values in the action columns, but infer_missing_actions is set to false. 
                      These actions will not be used to inform parameter estimation, but they will still be passed to the action model. 
                      Check that this is desired behaviour. This can especially be a problem for models which depend on their previous actions.
                      """
            end
            infer_missing_actions = SkipMissingActions()
        end
    else
        #If there are no missing actions
        if !any(ismissing, Matrix(data[!, collect(action_cols)]))
            if verbose
                @warn "infer_missing_actions is set to true, but there are no missing values in the action columns. Setting infer_missing_actions to false"
            end
            #Remove any potential Missing type
            disallowmissing!(data, collect(action_cols))
            infer_missing_actions = NoMissingActions()
        else
            infer_missing_actions = InferMissingActions()
        end
    end

    ## Decide if there are multiple actions ##
    if length(action_cols) == 1
        multiple_actions = SingleAction()
    else
        multiple_actions = MultipleActions()
    end

    ## Extract names and types ##
    #Extract from the action model
    parameter_names = parameters_to_estimate
    parameter_types =
        [action_model.parameters[parameter_name].type for parameter_name in parameter_names]
    state_names = keys(action_model.states)
    state_types = [action_model.states[state_name].type for state_name in state_names]
    observation_names = keys(action_model.observations)
    observation_types = [
        action_model.observations[observation_name].type for
        observation_name in observation_names
    ]
    action_names = keys(action_model.actions)
    action_types = [action_model.actions[action_name].type for action_name in action_names]
    action_dist_types =
        [action_model.actions[action_name].dist_type for action_name in action_names]

    #Extract action and observation types from the data
    observation_types_data = eltype.(eachcol(data[!, collect(observation_cols)]))
    action_types_data = eltype.(eachcol(data[!, collect(action_cols)]))

    #TODO: extract by running the model forward

    ## Run checks for the model specifications ##
    check_model(
        action_model,
        population_model,
        data,
        observation_cols,
        action_cols,
        grouping_cols,
        population_model_type;
        parameters_to_estimate,
        parameter_names = parameter_names,
        parameter_types = parameter_types,
        state_names = state_names,
        state_types = state_types,
        observation_names = observation_names,
        observation_types = observation_types,
        action_names = action_names,
        action_types = action_types,
        action_dist_types = action_dist_types,
    )


    ###s PREPARE DATA ###

    ## Initialize an agent ##
    agent_model = init_agent(action_model, save_history = false)

    ## Group data by sessions ##
    grouped_data = groupby(data, grouping_cols)

    ## Create IDs for each session ##
    session_ids = [
        join(
            [
                string(col_name) * id_column_separator * string(first(subdata)[col_name]) for col_name in grouping_cols
            ],
            id_separator,
        ) for subdata in grouped_data
    ]

    ## Extract observations and actions ##
    observations = Vector{Tuple{observation_types_data...}}[
        Tuple{observation_types_data...}.(
            eachrow(session_data[!, collect(observation_cols)]),
        ) for session_data in grouped_data
    ]
    actions = Vector{Tuple{action_types...}}[
        Tuple{action_types_data...}.(eachrow(session_data[!, collect(action_cols)])) for
        session_data in grouped_data
    ]

    ### CREATE MODEL ###
    ## Create the session model ##
    session_model = create_session_model(
        infer_missing_actions,
        multiple_actions,
        Val(check_parameter_rejections),
        actions,
    )

    ## Create a full model ##
    model = full_model(
        agent_model,
        parameters_to_estimate,
        population_model,
        session_model,
        session_ids,
        observations,
        actions,
    )

    return ModelFit(
        model = model,
        population_model_type = population_model_type,
        info = ModelFitInfo(
            estimated_parameter_names = parameters_to_estimate,
            session_ids = session_ids,
        ),
    )
end


####################################################################
### FUNCTION FOR DOING FULL AGENT AND STATISTICAL MODEL COMBINED ###
####################################################################
@model function full_model(
    agent_model::Agent,
    parameters_to_estimate::Vector{Symbol},
    population_model::DynamicPPL.Model,
    session_model::Function,
    session_ids::Vector{String},
    observations_per_session::Vector{Vector{O}},
    actions_per_session::Vector{Vector{A}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {
    O<:Tuple{Vararg{Any}},
    A<:Tuple{Vararg{Real}},
    TF,
    TI,
}

    #Generate session parameters with the population submodel
    parameters_per_session ~ to_submodel(population_model, false)

    #Generate behavior for each session
    i ~ to_submodel(
        session_model(
            agent_model,
            parameters_to_estimate,
            session_ids,
            parameters_per_session,
            observations_per_session,
            actions_per_session,
        ),
        false, #Do not add a prefix
    )
end






#########################################
#### FUNCTION FOR CHECKING THE MODEL ####
#########################################
function check_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    data::DataFrame,
    observation_cols::NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
    action_cols::NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
    grouping_cols::Vector{Symbol},
    population_model_type::AbstractPopulationModel,
    parameters_to_estimate::Vector{Symbol};
    parameter_names::Vector{Symbol},
    parameter_types::Vector{Type},
    state_names::Vector{Symbol},
    state_types::Vector{Type},
    observation_names::Vector{Symbol},
    observation_types::Vector{Type},
    action_names::Vector{Symbol},
    action_types::Vector{Type},
    action_dist_types::Vector{Type},
) where {observation_names,action_names}

    #Check that user-specified columns exist in the dataset
    if any(grouping_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified group columns that do not exist in the dataframe",
            ),
        )
    elseif any(values(observation_cols) .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified observation columns that do not exist in the dataframe",
            ),
        )
    elseif any(values(action_cols) .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified action columns that do not exist in the dataframe",
            ),
        )
    end

    #Check that observation and action column names exist in the action model
    for (observation_name_col, observation_col) in pairs(observation_cols)
        if !(observation_name_col in observation_names)
            throw(
                ArgumentError(
                    "The observation column $observation_col does not exist in the action model",
                ),
            )
        end
    end
    for (action_name_col, action_col) in pairs(action_cols)
        if !(action_name_col in action_names)
            throw(
                ArgumentError(
                    "The action column $action_col does not exist in the action model",
                ),
            )
        end
    end

    #Check whether observation and action columns are subtypes of what is specified in the action model
    for (action_col, (action_name, action)) in zip(action_cols, pairs(action_model.actions))
        if !(
            eltype(data[!, action_col]) <: action.type ||
            eltype(data[!, action_col]) <: Union{Missing,T} where {T<:action.type}
        )
            throw(
                ArgumentError(
                    "The action colum $action_col has type $(eltype(data[!, action_col])), but must be a subtype of the $action_name type specified in the action model: $(action.type)",
                ),
            )
        end
    end
    for (observation_col, (observation_name, observation)) in
        zip(observation_cols, pairs(action_model.observations))
        if !(eltype(data[!, observation_col]) <: observation.type)
            throw(
                ArgumentError(
                    "The observation column $observation_col has type $(eltype(data[!, observation_col])), but must be a subtype of the $observation_name type specified in the action model: $(observation.type)",
                ),
            )
        end
    end

    #Check whether there are NaN values in the action columns
    if any(isnan.(skipmissing(Matrix(data[!, collect(action_cols)]))))
        throw(ArgumentError("There are NaN values in the action columns"))
    end
end
