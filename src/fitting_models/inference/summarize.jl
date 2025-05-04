#######################################
##### SUMMARIZE SESSION PARAMETERS ####
#######################################
function Turing.summarize(
    session_parameters::SessionParameters,
    summary_function::Function = median;
)
    #Extract sessions and parameters
    session_parameters = session_parameters.value
    sessions = session_parameters.axes[1]
    parameters = session_parameters.axes[2]

    #Construct grouping column names
    grouping_cols = [
        Symbol(first(split(i, id_column_separator))) for
        i in split(string(first(sessions)), id_separator)
    ]

    # Initialize an empty DataFrame
    df = DataFrame(Dict(Symbol(parameter) => Float64[] for parameter in parameters))
    #Add grouping colnames
    for column_name in grouping_cols
        df[!, column_name] = String[]
    end

    # Populate the DataFrame with summarized values
    for session_id in sessions
        row = Dict()
        for parameter in parameters
            # Extract the values for the current session and parameter across samples and chains
            values = session_parameters[session_id, parameter, :, :]
            # Calculate the median value
            summarized_value = summary_function(values)
            # Add the median value to the row
            row[Symbol(parameter)] = summarized_value
        end

        #Split session ids
        split_session_ids = split(string(session_id), id_separator)
        #Add them to the row
        for (session_id_part, column_name) in zip(split_session_ids, grouping_cols)
            row[column_name] = string(split(session_id_part, id_column_separator)[2])
        end

        # Add the row to the DataFrame
        push!(df, row)
    end

    # Reorder the columns to have session id's as the first columns
    select!(df, grouping_cols, names(df)[1:end-length(grouping_cols)]...)

    return df
end


#######################################
##### SUMMARIZE STATE TRAJECTORIES ####
#######################################
function Turing.summarize(
    state_trajectories::StateTrajectories{T},
    summary_function::Function = median,
) where T

    #Extract sessions ids and state trajectories
    state_names = state_trajectories.state_names
    session_ids = state_trajectories.session_ids
    state_trajectories = state_trajectories.value

    # Initialize an empty vector to store summarized values
    summarized_values = Vector{Matrix{Union{Missing,Float64}}}()
    timestep_cols = Vector{Int}()
    session_id_cols = Vector{Matrix{String}}()

    #For each session
    for (session_id, session_state_trajectories) in zip(session_ids, state_trajectories)

        #Summarize across samples
        session_summarized_values =
            summarize_samples.(
                eachslice(session_state_trajectories, dims = (1, 2)),
                summary_function,
            )

        #Add to the vector
        push!(summarized_values, session_summarized_values)

        #Extract and add timesteps
        timesteps = collect(session_state_trajectories.axes[1])
        append!(timestep_cols, timesteps)

        #Split session id
        split_session_ids = map(
            i -> String(split(i, id_column_separator)[2]),
            split(string(session_id), id_separator),
        )
        #Repeat it for each timestep
        split_session_ids =
            permutedims(hcat(repeat([split_session_ids], length(timesteps))...))
        push!(session_id_cols, split_session_ids)
    end

    #Construct grouping column names
    grouping_colnames = [
        Symbol(first(split(i, id_column_separator))) for
        i in split(string(first(session_ids)), id_separator)
    ]

    #Create final dataframe
    output_df = hcat(
        DataFrame(vcat(session_id_cols...), grouping_colnames),
        DataFrame(timestep = vcat(timestep_cols...)),
        DataFrame(vcat(summarized_values...), collect(state_names)),
    )

    return output_df
end

function summarize_samples(array::A, summary_function::Function) where {A<:AxisArray}
    # Handle missing values
    if all(ismissing, array)
        return missing
    elseif any(ismissing, array)
        return summary_function(skipmissing(array))
    else
        return summary_function(array)
    end
end


