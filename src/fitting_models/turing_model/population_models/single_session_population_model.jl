#############################################
### POPULATION MODEL FOR A SINGLE SESSION ###
#############################################
function create_model(
    agent::Agent,
    prior::Dict{String,D},
    inputs::II,
    actions::AA;
    verbose::Bool = true,
    kwargs...,
) where {
    D<:Distribution,
    I<:Union{<:Any, NTuple{N, <:Any} where N},
    II<:Vector{I},
    A<:Union{<:Real, NTuple{N, <:Real} where N},
    AA<:Vector{A},
}
    
    #Check population_model
    check_population_model(
        SingleSessionPopulationModel(),
        agent,
        prior,
        inputs,
        actions,
        verbose;
        kwargs...,
    )

    #Get number of 
    n_inputs = length(first(inputs))    
    n_actions = length(first(actions))

    #Create column names
    input_cols = map(x -> Symbol("input_$x"), 1:n_inputs)
    action_cols = map(x -> Symbol("action_$x"), 1:n_actions)

    #Create dataframe of the inputs and actions
    data = hcat(
        DataFrame(NamedTuple{Tuple(input_cols)}.(inputs)),
        DataFrame(NamedTuple{Tuple(action_cols)}.(actions)),
    )

    #Add grouping column
    grouping_cols = "session"
    data[!, grouping_cols] .= 1

    #Create an independent_population_model with the single session
    return create_model(
        agent,
        prior,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        verbose = verbose,
        population_model_type = SingleSessionPopulationModel(),
        kwargs...,
    )
end


##############################################
####### CHECKS FOR THE POPULATION MODEL ######
##############################################
function check_population_model(
    model_type::SingleSessionPopulationModel,
    agent::Agent,
    prior::Dict{String,D},
    inputs::II,
    actions::AA,
    verbose::Bool;
    kwargs...,
) where {
    D<:Distribution,
    I<:Union{<:Any, NTuple{N, <:Any} where N},
    II<:Vector{I},
    A<:Union{<:Real, NTuple{N, <:Real} where N},
    AA<:Vector{A},
}

    if length(inputs) != length(actions)
        throw(ArgumentError("The inputs and actions vectors must have the same length."))
    end

    if !all(y->y==length.(inputs)[1],length.(inputs))
        throw(ArgumentError("All tuples in the inputs vector must have the same length."))
    end

    if !all(y->y==length.(actions)[1],length.(actions))
        throw(ArgumentError("All tuples in the actions vector must have the same length."))
    end
end
