###########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A STATISTICAL MODEL ###
###########################################################################################################
function create_model(
    agent::Agent,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3} = Vector{String}(),
    parameter_names::Vector{String},
    infer_missing_actions::Bool = false,
    check_parameter_rejections::Bool = false,
    population_model_type::AbstractPopulationModel = CustomPopulationModel(),
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
    input_cols = Symbol.(input_cols)

    if !(action_cols isa Vector)
        action_cols = [action_cols]
    end
    action_cols = Symbol.(action_cols)

    if !(grouping_cols isa Vector)
        grouping_cols = [grouping_cols]
    end
    grouping_cols = Symbol.(grouping_cols)

    ## Check whether to skip or infer missing data ##
    if !infer_missing_actions
        #If there are no missing actions
        if !any(ismissing, Matrix(data[!, action_cols]))
            #Remove any potential Missing type
            disallowmissing!(data, action_cols)
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
        if !any(ismissing, Matrix(data[!, action_cols]))
            if verbose
                @warn "infer_missing_actions is set to true, but there are no missing values in the action columns. Setting infer_missing_actions to false"
            end
            #Remove any potential Missing type
            disallowmissing!(data, action_cols)
            infer_missing_actions = nothing
        else
            infer_missing_actions = InferMissingActions()
        end
    end

    # Run checks for the model specifications ##
    check_model(
        agent_model,
        population_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        population_model_type = population_model_type,
        verbose = verbose,
    )

    ## EXTRACT DATA ##
    #Group data by sessions
    grouped_data = groupby(data, grouping_cols)

    #Create IDs for each session
    session_ids = [
        Symbol(
            join(
                [
                    string(col_name) *
                    id_column_separator *
                    string(first(subdata)[col_name]) for col_name in grouping_cols
                ],
                id_separator,
            ),
        ) for subdata in grouped_data
    ]

    ## Extract inputs and actions ##
    if length(input_cols) == 1
        #Inputs are a vector of vectors of <:Any
        inputs = [Vector(agent_data[!, first(input_cols)]) for agent_data in grouped_data]
    else
        #Extract action types
        input_types = eltype.(eachcol(data[!, input_cols]))
        #Inputs are a vector of vectors of tuples of <:Any
        inputs = [
            Tuple{input_types...}.(eachrow(agent_data[!, input_cols])) for
            agent_data in grouped_data
        ]

        #Retype actions
        inputs = Vector{Vector{Tuple{input_types...}}}(inputs)
    end

    if length(action_cols) == 1
        #Actions are a vector of vectors of <:Real
        actions = [Vector(agent_data[!, first(action_cols)]) for agent_data in grouped_data]
        multiple_actions = false
    else
        #Extract action types
        action_types = eltype.(eachcol(data[!, action_cols]))
        #Actions are a vector of vectors of tuples of <:Real
        actions = [
            Tuple{action_types...}.(eachrow(agent_data[!, action_cols])) for
            agent_data in grouped_data
        ]

        #Retype actions
        actions = Vector{Vector{Tuple{action_types...}}}(actions)

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
    return full_model(
        agent_model,
        parameter_names,
        population_model,
        session_model,
        session_ids,
        inputs,
        actions,
    )
end



####################################################################
### FUNCTION FOR DOING FULL AGENT AND STATISTICAL MODEL COMBINED ###
####################################################################
@model function full_model(
    agent_model::Agent,
    parameter_names::Vector{String},
    population_model::DynamicPPL.Model,
    session_model::Function,
    session_ids::Vector{Symbol},
    inputs_per_session::Vector{Vector{II}},
    actions_per_session::Vector{Vector{AA}},
) where {I<:Any,II<:Union{I,Tuple},A<:Union{<:Real,Missing},AA<:Union{A,<:Tuple}}

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

    #Return the session parameters
    return parameters_per_session
end






#########################################
#### FUNCTION FOR CHECKING THE MODEL ####
#########################################
function check_model(
    agent::Agent,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    population_model_type::AbstractPopulationModel,
    verbose::Bool = true,
) where {T1<:Union{String,Symbol},T2<:Union{String,Symbol},T3<:Union{String,Symbol}}

    #Check that user-specified columns exist in the dataset
    if any(grouping_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified group columns that do not exist in the dataframe",
            ),
        )
    elseif any(input_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified input columns that do not exist in the dataframe",
            ),
        )
    elseif any(action_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified action columns that do not exist in the dataframe",
            ),
        )
    end

    #Check whether the action columns are of the correct type
    if any(!((data[!, action_cols] isa Vector{<:Real})))
        throw(
            ArgumentError(
                "The action columns must be of type Vector{<:Real}",
            ),
        )
    end

    #Check whether there are NaN values in the action columns
    if any(isnan.(data[!, action_cols]))
        throw(
            ArgumentError(
                "There are NaN values in the action columns",
            ),
        )
    end
end
