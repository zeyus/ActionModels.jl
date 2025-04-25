##########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A POPULATION MODEL ###
##########################################################################################################
function create_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{
        NamedTuple{input_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    grouping_cols::Union{Vector{Symbol},Symbol} = Vector{Symbol}(),
    parameter_names::Vector{Symbol},
    infer_missing_actions::Bool = false,
    check_parameter_rejections::Bool = false,
    population_model_type::AbstractPopulationModel = CustomPopulationModel(),
    verbose::Bool = true,
) where {input_names,action_names}

    ### PRE-SETUP CHECKS ###
    #Check that input cols and action cols are the same length as the inputs and actions in the action model
    if length(input_cols) != length(action_model.observations)
        throw(
            ArgumentError(
                "The number of input columns does not match the number of inputs in the action model",
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

    ### SETUP ###
    ## Change columns to the correct format ##
    #Make single action and input columns into vectors
    if input_cols isa Symbol
        input_cols = [input_cols]
    end
    if action_cols isa Symbol
        action_cols = [action_cols]
    end
    #Make sure that input_cols and action_cols are named tuples
    if input_cols isa Vector
        input_cols = NamedTuple{keys(action_model.observations)}(input_cols)
        if verbose && length(input_cols) > 1
            @warn "Mappings from action model inputs to input columns not provided. Using the order from the action model: $(input_cols)"
        end
    end
    if action_cols isa Vector
        action_cols = NamedTuple{keys(action_model.actions)}(action_cols)
        if verbose && length(action_cols) > 1
            @warn "Mappings from action model actions to action columns not provided. Using the order from the action model: $(action_cols)"
        end
    end
    #Order input action columns to match the action model
    input_cols = NamedTuple(
        input_name => input_cols[input_name] for input_name in keys(action_model.observations)
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
            infer_missing_actions = nothing
        else
            if verbose
                @warn """
                      There are missing values in the action columns, but infer_missing_actions is set to false. 
                      These actions will not be used for fitting, but they will still be passed to the action model. 
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
            infer_missing_actions = nothing
        else
            infer_missing_actions = InferMissingActions()
        end
    end

    ## Run checks for the model specifications ##
    check_model(
        action_model,
        population_model,
        data,
        input_cols,
        action_cols,
        grouping_cols,
        parameter_names,
        population_model_type,
    )

    ## Initialize an agent ##
    agent_model = init_agent(action_model, save_history = false)

    ## EXTRACT DATA ##
    #Group data by sessions
    grouped_data = groupby(data, grouping_cols)

    #Create IDs for each session
    session_ids = [
        join(
            [
                string(col_name) * id_column_separator * string(first(subdata)[col_name]) for col_name in grouping_cols
            ],
            id_separator,
        ) for subdata in grouped_data
    ]

    ## Extract inputs and actions ## #TODO: make this work with the now namedtuple formatted cols
    if length(input_cols) == 1
        #Inputs are a vector of vectors of <:Any
        inputs = [Vector(session_data[!, first(input_cols)]) for session_data in grouped_data]
    else
        #Extract input types
        input_types = eltype.(eachcol(data[!, collect(input_cols)]))
        #Inputs are a vector of vectors of tuples of <:Any
        inputs = Vector{Tuple{input_types...}}[
            Tuple{input_types...}.(eachrow(session_data[!, collect(input_cols)])) for
            session_data in grouped_data
        ]
    end

    if length(action_cols) == 1
        #Actions are a vector of vectors of <:Real
        actions = [Vector(session_data[!, first(action_cols)]) for session_data in grouped_data]
        multiple_actions = false
    else
        #Extract action types
        action_types = eltype.(eachcol(data[!, collect(action_cols)]))
        #Actions are a vector of vectors of tuples of <:Real
        actions = Vector{Tuple{action_types...}}[
            Tuple{action_types...}.(eachrow(session_data[!, collect(action_cols)])) for
            session_data in grouped_data
        ]

        multiple_actions = true
    end

    ## SELECT SESSION MODEL ##
    session_model = create_session_model(
        infer_missing_actions,
        Val(multiple_actions),
        Val(check_parameter_rejections),
        actions,
    )

    #Create a full model combining the agent model and the statistical model
    model = full_model(
        agent_model,
        parameter_names,
        population_model,
        session_model,
        session_ids,
        inputs,
        actions,
    )

    return ModelFit(
        model = model,
        population_model_type = population_model_type,
        info = ModelFitInfo(parameter_names = parameter_names, session_ids = session_ids),
    )
end



####################################################################
### FUNCTION FOR DOING FULL AGENT AND STATISTICAL MODEL COMBINED ###
####################################################################
@model function full_model(
    agent_model::Agent,
    parameter_names::Vector{Symbol},
    population_model::DynamicPPL.Model,
    session_model::Function,
    session_ids::Vector{String},
    inputs_per_session::Vector{Vector{II}},
    actions_per_session::Vector{Vector{AA}},
) where {I<:Any,II<:Union{I,Tuple{Vararg{I}}},A<:Union{<:Real,Missing},AA<:Union{A,<:Tuple{Vararg{Union{Missing,A}}}}}

    #Generate session parameters with the population submodel
    parameters_per_session ~ to_submodel(population_model, false)

    #Generate behavior for each session
    i ~ to_submodel(
        session_model(
            agent_model,
            parameter_names,
            session_ids,
            parameters_per_session,
            inputs_per_session,
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
    input_cols::NamedTuple{input_names,<:Tuple{Vararg{Symbol}}},
    action_cols::NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
    grouping_cols::Vector{Symbol},
    parameter_names::Vector{Symbol},
    population_model_type::AbstractPopulationModel,
) where {input_names,action_names}
    #Check that user-specified columns exist in the dataset
    if any(grouping_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified group columns that do not exist in the dataframe",
            ),
        )
    elseif any(values(input_cols) .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified input columns that do not exist in the dataframe",
            ),
        )
    elseif any(values(action_cols) .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified action columns that do not exist in the dataframe",
            ),
        )
    end

    #Check that input and action column names exist in the action model
    for (input_name_data, input_col) in pairs(input_cols)
        if !(input_name_data in keys(action_model.observations))
            throw(
                ArgumentError(
                    "The input column $input_col does not exist in the action model",
                ),
            )
        end
    end
    for (action_name_data, action_col) in pairs(action_cols)
        if !(action_name_data in keys(action_model.actions))
            throw(
                ArgumentError(
                    "The action column $action_col does not exist in the action model",
                ),
            )
        end
    end

    #Check whether input and action columns are subtypes of what is specified in the action model
    for (action_col, (action_name, action)) in zip(action_cols, pairs(action_model.actions))
        if !(eltype(data[!, action_col]) <: action.type || eltype(data[!, action_col]) <: Union{Missing, T} where T<:action.type)
            throw(
                ArgumentError(
                    "The action colum $action_col has type $(eltype(data[!, action_col])), but must be a subtype of the $action_name type specified in the action model: $(action.type)",
                ),
            )
        end
    end
    for (input_col, (input_name, input)) in zip(input_cols, pairs(action_model.observations))
        if !(eltype(data[!, input_col]) <: input.type)
            throw(
                ArgumentError(
                    "The input column $input_col has type $(eltype(data[!, input_col])), but must be a subtype of the $input_name type specified in the action model: $(input.type)",
                ),
            )
        end
    end

    #Check whether there are NaN values in the action columns
    if any(isnan.(skipmissing(Matrix(data[!, collect(action_cols)]))))
        throw(ArgumentError("There are NaN values in the action columns"))
    end
end
