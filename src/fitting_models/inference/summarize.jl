#######################################
##### SUMMARIZE SESSION PARAMETERS ####
#######################################
function Turing.summarize(
    session_parameters::SessionParameters,
    summary_function::Function = median;
)
    #Extract sessions and parameters
    session_ids = session_parameters.session_ids
    estimated_parameter_names = session_parameters.estimated_parameter_names
    parameter_types = session_parameters.parameter_types
    session_parameters = session_parameters.value

    #Construct grouping column names
    grouping_cols = [
        Symbol(first(split(i, id_column_separator))) for
        i in split(string(first(session_ids)), id_separator)
    ]


    ## Initialize an empty DataFrame with appropriate columns. ##
    df_cols = Dict{String,Any}()
    # For each ParameterDependentState
    for parameter in parameter_names
        #If its first sample is an AbstractArray
        sample_parameter = first(session_parameters[parammeter][1])
        if sample_parameter isa AbstractArray
            #Make a column for each Cartesian index
            for I in CartesianIndices(size(sample_parameter))
                col_name = "$(parameter)$(Tuple(I))"
                df_cols[col_name] = eltype(parameter_types[parameter])[]
            end
        else
            #Oherwise, just make a column for the state
            df_cols[parameter] = parameter_types[parameter][]
        end
    end

    #Create dataframe
    df = DataFrame(df_cols)

    #Add grouping colnames
    for column_name in grouping_cols
        df[!, column_name] = String[]
    end

    ## Populate the DataFrame with summarized values ##
    row = Dict()
    #Loop over sessions and parameters
    for session_id in session_ids
        for parameter in estimated_parameter_names

            #Split session ids
            split_session_ids = split(string(session_id), id_separator)
            #Add them to the row
            for (session_id_part, column_name) in zip(split_session_ids, grouping_cols)
                row[column_name] = string(split(session_id_part, id_column_separator)[2])
            end

            # Extract the values for the current session and parameter across samples and chains
            samples = session_parameters[parameter][Symbol(session_id)]

            #If the samples are AbstractArrays
            if first(samples) isa AbstractArray

                #For each Cartesian index
                for I in CartesianIndices(size(first(samples)))
                    #Unpack that index and summarize the samples
                    summarized_value =
                        summarize_samples(map(s -> s[I], samples), summary_function)
                    row["$(parameter)$(Tuple(I))"] = summarized_value
                end
            else
                summarized_value = summarize_samples(samples, summary_function)
                row[parameter] = summarized_value
            end
        end

        # Add the row to the DataFrame
        push!(df, row)
    end

    # Reorder the columns to have session id's as the first columns
    select!(df, grouping_cols, names(df)[1:(end-length(grouping_cols))]...)

    return df
end


#######################################
##### SUMMARIZE STATE TRAJECTORIES ####
#######################################
function Turing.summarize(
    state_trajectories::StateTrajectories,
    summary_function::Function = median,
)

    #Extract sessions ids and state trajectories
    state_names = state_trajectories.state_names
    session_ids = state_trajectories.session_ids
    state_types = state_trajectories.state_types
    state_trajectories = state_trajectories.value

    #Construct grouping column names
    grouping_cols = [
        first(split(i, id_column_separator)) for
        i in split(string(first(session_ids)), id_separator)
    ]

    ## Initialize an empty DataFrame with appropriate columns. ##
    df_cols = Dict{String,Any}()
    # For each state
    for state in state_names
        #If its first sample is an AbstractArray
        sample_state = first(state_trajectories[state][1])
        if sample_state isa AbstractArray
            #Make a column for each Cartesian index
            for I in CartesianIndices(size(sample_state))
                col_name = "$(state)$(Tuple(I))"
                df_cols[col_name] = eltype(state_types[state])[]
            end
        else
            #Oherwise, just make a column for the state
            df_cols[state] = state_types[state][]
        end
    end
    #Create dataframe
    df = DataFrame(df_cols)

    #Add grouping columns and the timestep column.
    for column_name in grouping_cols
        df[!, column_name] = String[]
    end
    df[!, "timestep"] = Int[]


    # Populate the DataFrame with summarized values
    row = Dict()
    #Loop for sessions and states
    for session_id in session_ids
        for state in state_names

            #Split session ids
            split_session_ids = split(string(session_id), id_separator)
            #Add them to the row
            for (session_id_part, column_name) in zip(split_session_ids, grouping_cols)
                row[column_name] = string(split(session_id_part, id_column_separator)[2])
            end

            # Extract the values for the current session and state across samples, chains and timesteps
            values = state_trajectories[state][Symbol(session_id)]

            #Loop over timesteps
            for timestep = 1:size(values, 3)

                # Add the timestep to the row
                row["timestep"] = timestep-1

                # Extract the values for the current session and state across samples and chains
                samples = values[:, :, timestep]

                #If the samples are AbstractArrays
                if first(samples) isa AbstractArray

                    #For each Cartesian index
                    for I in CartesianIndices(size(first(samples)))
                        #Unpack that index and summarize the samples
                        summarized_value =
                            summarize_samples(map(s -> s[I], samples), summary_function)
                        row["$(state)$(Tuple(I))"] = summarized_value
                    end
                else
                    summarized_value = summarize_samples(samples, summary_function)
                    row[state] = summarized_value
                end

                # Add the row to the DataFrame
                push!(df, row)
            end
        end
    end

    # Reorder the columns to have session id's as the first columns
    select!(df, grouping_cols, "timestep", names(df)[1:(end-length(grouping_cols)-1)]...)

    return df
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





#     # Populate the DataFrame with summarized values
#     for session_id in session_ids
#         for state in state_names
#             # Prepare a new row for each timestep.
#             split_session_ids = split(string(session_id), id_separator)
#             base_row = Dict{Symbol, Any}()
#             for (session_id_part, column_name) in zip(split_session_ids, grouping_cols)
#                 base_row[column_name] = string(split(session_id_part, id_column_separator)[2])
#             end
#             # Extract the values for the current session and state.
#             values = state_trajectories[state][Symbol(session_id)]
#             for timestep in 1:size(values, 3)
#                 row = copy(base_row)
#                 row[:timestep] = timestep - 1
#                 samples = values[:, :, timestep]
#                 if eltype(samples) <: AbstractArray
#                     # Unpack inner arrays.
#                     first_inner = samples[1,1]
#                     inner_dims = size(first_inner)
#                     for I in CartesianIndices(inner_dims)
#                         inner_values = [s[I] for s in samples]
#                         summarized_value = summary_function(inner_values)
#                         row[Symbol("$(state)_$(I)")] = summarized_value
#                     end
#                 else
#                     summarized_value = summary_function(samples)
#                     row[Symbol(state)] = summarized_value
#                 end
#                 push!(df, row)
#             end