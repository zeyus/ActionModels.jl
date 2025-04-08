function Turing.summarize(
    session_parameters::AxisArray,
    sink::T = DataFrame,
    summary_function::Function = median,
) where {T<:Union{Type{Dict},Type{DataFrame}}}

    summarize(session_parameters, summary_function, sink)

end





#######################################
##### SUMMARIZE SESSION PARAMETERS ####
#######################################
function Turing.summarize(
    session_parameters::AxisArray{
        Float64,
        4,
        Array{Float64,4},
        Tuple{
            Axis{:session,Vector{String}},
            Axis{:parameter,Vector{String}},
            Axis{:sample,UnitRange{Int64}},
            Axis{:chain,UnitRange{Int64}},
        },
    },
    sink::Type{DataFrame},
    summary_function::Function = median;
    promote::Bool = false,
)

    #Extract sessions and parameters
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


#########################################################
####### VERSION WHICH GENERATES A DICTIONARY INSTEAD ####
#########################################################
function Turing.summarize(
    session_parameters::AxisArray{
        Float64,
        4,
        Array{Float64,4},
        Tuple{
            Axis{:session,Vector{String}},
            Axis{:parameter,Vector{String}},
            Axis{:sample,UnitRange{Int64}},
            Axis{:chain,UnitRange{Int64}},
        },
    },
    output_type::Type{Dict},
    summary_function::Function = median,
)

    #Extract sessions and parameters
    sessions = session_parameters.axes[1]
    parameters = session_parameters.axes[2]

    # Initialize an empty dictionary
    estimates_dict = Dict{String,Dict{String,Float64}}()

    # Populate the dictionary with summarized values
    for (i, session) in enumerate(sessions)
        session_dict = Dict{String,Float64}()
        for (j, parameter) in enumerate(parameters)
            # Extract the values for the current session and parameter across samples and chains
            values = session_parameters[session, parameter, :, :]
            # Calculate the median value
            summarized_value = summary_function(values)
            # Add the median value to the session's dictionary
            session_dict[parameter] = summarized_value
        end
        # Add the session's dictionary to the main dictionary
        estimates_dict[session] = session_dict
    end

    return estimates_dict
end








#######################################
##### SUMMARIZE STATE TRAJECTORIES ####
#######################################
function Turing.summarize(
    state_trajectories::AxisArrays.AxisArray{
        Union{Missing,Float64},
        5,
        Array{Union{Missing,Float64},5},
        Tuple{
            AxisArrays.Axis{:session,Vector{String}},
            AxisArrays.Axis{:state,Vector{String}},
            AxisArrays.Axis{:timestep,UnitRange{Int64}},
            AxisArrays.Axis{:sample,UnitRange{Int64}},
            AxisArrays.Axis{:chain,UnitRange{Int64}},
        },
    },
    summary_function::Function = median,
)

    #Extract sessions and parameters
    sessions = state_trajectories.axes[1]
    states = state_trajectories.axes[2]
    timesteps = state_trajectories.axes[3]

    #Construct grouping column names
    grouping_cols = [
        Symbol(first(split(i, id_column_separator))) for
        i in split(string(first(sessions)), id_separator)
    ]

    # Initialize an empty DataFrame with the states, the grouping columns and the timestep
    df = DataFrame(Dict(begin
        #Join tuples
        if state isa Tuple
            state = join(state, tuple_separator)
        end
        state => Float64[]
    end for state in states))
    for column_name in grouping_cols
        df[!, column_name] = String[]
    end
    df[!, :timestep] = Int[]


    # Populate the DataFrame with median values
    for session_id in sessions

        for timestep in timesteps
            row = Dict()

            for state in states
                # Extract the state for the current session and state, at the current timestep
                values = state_trajectories[session_id, state, timestep+1, :, :]
                # Calculate the point estimate
                median_value = summary_function(values)

                #Join tuples
                if state isa Tuple
                    state = join(state, tuple_separator)
                end

                # Add the value to the row
                row[state] = median_value
            end

            #Split session ids
            split_session_ids = split(string(session_id), id_separator)
            #Add them to the row
            for (session_id_part, column_name) in zip(split_session_ids, grouping_cols)
                row[column_name] = string(split(session_id_part, id_column_separator)[2])
            end

            #Add the timestep to the row
            row[:timestep] = timestep

            # Add the row to the DataFrame
            push!(df, row, promote = true)
        end
    end

    # Reorder the columns to have session_id as the first column
    select!(
        df,
        vcat(grouping_cols, [:timestep]),
        names(df)[1:end-(length(grouping_cols)+1)]...,
    )

    return df
end
