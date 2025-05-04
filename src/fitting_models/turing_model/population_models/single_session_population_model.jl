#############################################
### POPULATION MODEL FOR A SINGLE SESSION ###
#############################################
function create_model(
    action_model::ActionModel,
    prior::NamedTuple{prior_names, <:Tuple{Vararg{Distribution}}},
    observations::Vector{OO},
    actions::Vector{AA};
    verbose::Bool = true,
    kwargs...,
) where {
    O <:Any,
    OO<:Union{O,Tuple{Vararg{O}}},
    A <:Union{Missing, Real},
    AA<:Union{A, Tuple{Vararg{A}}},
    prior_names,
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

    #Check if the observations and actions are tuples
    multiple_observations = OO <: Tuple
    multiple_actions = AA <: Tuple

    #Get number of observations and actions
    if !multiple_observations
        n_observations = 1
    else
        n_observations = length(first(observations))   
    end
    if !multiple_actions
        n_actions = 1
    else
        n_actions = length(first(actions))
    end

    #Create column names
    observation_cols = map(x -> Symbol("observation_$x"), 1:n_observations)
    action_cols = map(x -> Symbol("action_$x"), 1:n_actions)

    #Make observations and actions into tuples of vectors
    observations = evert(observations)
    actions = evert(actions)

    #Create dataframe of the observations and actions
    data = hcat(
        DataFrame(NamedTuple{Tuple(observation_cols)}(observations)),
        DataFrame(NamedTuple{Tuple(action_cols)}(actions)),
    )

    #Add grouping column
    grouping_cols = :session
    data[!, grouping_cols] .= :single_session

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
    prior::NamedTuple{prior_names, <:Tuple{Vararg{Distribution}}},
    observations::II,
    actions::AA,
    verbose::Bool;
    kwargs...,
) where {
    prior_names,
    II <: Vector,
    AA <: Vector,
}

    if length(observations) != length(actions)
        throw(ArgumentError("The observations and actions vectors must have the same length."))
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