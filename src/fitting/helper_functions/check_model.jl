###############################################
#### FUNCTION FOR CHECKING A CREATED MODEL ####
###############################################
function check_model(
    agent::Agent,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    verbose::Bool = true,
) where {T1<:Union{String,Symbol},T2<:Union{String,Symbol},T3<:Union{String,Symbol}}

    #Run the check of the statistical model    
    check_population_model(population_model.args...; verbose = verbose, agent = agent)

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
