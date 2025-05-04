#########################################################
### SIMPLE POPULATION MODEL WITH INDEPENDENT SESSIONS ###
#########################################################
function create_model(
    action_model::ActionModel,
    prior::NamedTuple{prior_names, <:Tuple{Vararg{Distribution}}},
    data::DataFrame;
    observation_cols::Union{
        NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    grouping_cols::Union{Vector{Symbol},Symbol} = Vector{Symbol}(),
    population_model_type::Union{IndependentPopulationModel,SingleSessionPopulationModel} = IndependentPopulationModel(),
    verbose::Bool = true,
    kwargs...,
) where {
    prior_names,
    observation_names, 
    action_names
}

    #Check population_model
    check_population_model(
        IndependentPopulationModel(),
        action_model,
        prior,
        data,
        observation_cols,
        action_cols,
        grouping_cols,
        verbose;
        kwargs...,
    )

    #Get number of sessions
    n_sessions = length(groupby(data, grouping_cols))

    #Get the names of the estimated parameters
    parameters_to_estimate = keys(prior)

    #Create a filldist for each parameter
    priors_per_parameter = Tuple([
        filldist(getproperty(prior, param_name), n_sessions) for param_name in prior_names
    ])

    #Create a statistical model where the agents are independent and sampled from the same prior
    population_model = independent_population_model(priors_per_parameter, parameters_to_estimate)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        action_model,
        population_model,
        data;
        observation_cols = observation_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        parameters_to_estimate = parameters_to_estimate,
        population_model_type = population_model_type,
        kwargs...,
    )
end

#Turing model for sampling all sessions for all parameters
@model function independent_population_model(
    priors_per_parameter::T,
    parameters_to_estimate::Tuple{Vararg{Symbol}},
) where {T<:Tuple}

    sampled_parameters = Tuple(
        i ~ to_submodel(
            prefix(sample_parameters_all_session(prior), parameter_name),
            false,
        ) for (prior, parameter_name) in zip(priors_per_parameter, parameters_to_estimate)
    )

    return zip(sampled_parameters...)
end

#Turing submodel for sampling all sessions for a single parameter
@model function sample_parameters_all_session(prior)

    session ~ prior

    return session
end



##############################################
####### CHECKS FOR THE POPULATION MODEL ######
##############################################
function check_population_model(
    model_type::IndependentPopulationModel,
    action_model::ActionModel,
    prior::NamedTuple{prior_names, <:Tuple{Vararg{Distribution}}},
    data::DataFrame,
    observation_cols::Union{
        NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    grouping_cols::Union{Vector{Symbol},Symbol},
    verbose::Bool;
    kwargs...,
) where {prior_names, observation_names, action_names}
    #If there are no parameters to sample
    if length(prior) == 0
        #Throw an error
        throw(ArgumentError("There are no parameters in the prior."))
    end
end