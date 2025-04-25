#############################################
### POPULATION MODEL FOR A SINGLE SESSION ###
#############################################
function create_model(
    action_model::ActionModel,
    prior::Dict{Symbol,D},
    observations::II,
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
        action_model,
        prior,
        observations,
        actions,
        verbose;
        kwargs...,
    )

    #Get number of 
    n_observations = length(first(observations))    
    n_actions = length(first(actions))

    #Create column names
    observation_cols = map(x -> Symbol("observation_$x"), 1:n_observations)
    action_cols = map(x -> Symbol("action_$x"), 1:n_actions)

    #Create dataframe of the observations and actions
    data = hcat(
        DataFrame(NamedTuple{Tuple(observation_cols)}.(observations)),
        DataFrame(NamedTuple{Tuple(action_cols)}.(actions)),
    )

    #Add grouping column
    grouping_cols = :session
    data[!, grouping_cols] .= 1

    #Create an independent_population_model with the single session
    return create_model(
        action_model,
        prior,
        data;
        observation_cols = observation_cols,
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
    action_model::ActionModel,
    prior::Dict{Symbol,D},
    observations::II,
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

    if length(observations) != length(actions)
        throw(ArgumentError("The observations and actions vectors must have the same length."))
    end

    if !all(y->y==length.(observations)[1],length.(observations))
        throw(ArgumentError("All tuples in the observations vector must have the same length."))
    end

    if !all(y->y==length.(actions)[1],length.(actions))
        throw(ArgumentError("All tuples in the actions vector must have the same length."))
    end
end




########################################
####### DEFAULT PLOTTING FUNCTION ######
########################################

#Plotting a ModelFit just plots the session parameters
@recipe function f(modelfit::ModelFit{SingleSessionPopulationModel}; plot_prior = true) #::Bool

    #Get session parameters
    posterior_parameters = get_session_parameters!(modelfit, :posterior)
    if plot_prior == true
        prior_parameters = get_session_parameters!(modelfit, :prior)
    else
        prior_parameters = nothing
    end

    #Get the session id
    session_id = modelfit.info.session_ids[1]

    #Make the standard plot for just that session
    plot(posterior_parameters, session_id; prior_parameters = prior_parameters)

end