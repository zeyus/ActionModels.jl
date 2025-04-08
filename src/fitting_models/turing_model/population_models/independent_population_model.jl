#########################################################
### SIMPLE POPULATION MODEL WITH INDEPENDENT SESSIONS ###
#########################################################
struct IndependentPopulationModel <: AbstractPopulationModel end

function create_model(
    agent::Agent,
    prior::Dict{String,D},
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T2},
    grouping_cols::Union{Vector{T3},T3} = Vector{String}(),
    population_model_type::Union{IndependentPopulationModel,SingleSessionPopulationModel} = IndependentPopulationModel(),
    verbose::Bool = true,
    kwargs...,
) where {
    D<:Distribution,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
}

    #Check population_model
    check_population_model(
        IndependentPopulationModel(),
        agent,
        prior,
        data,
        input_cols,
        action_cols,
        grouping_cols,
        verbose;
        kwargs...,
    )

    #Get number of sessions
    n_sessions = length(groupby(data, grouping_cols))

    #Get the names of the estimated parameters
    parameter_names = collect(keys(prior))

    #Create a filldist for each parameter
    priors_per_parameter = Tuple([
        filldist(prior[parameter_name], n_sessions) for parameter_name in parameter_names
    ])

    #Create a statistical model where the agents are independent and sampled from the same prior
    population_model = independent_population_model(priors_per_parameter, parameter_names)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        agent,
        population_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        parameter_names = parameter_names,
        population_model_type = population_model_type,
        kwargs...,
    )
end

#Turing model for sampling all sessions for all parameters
@model function independent_population_model(
    priors_per_parameter::T,
    parameter_names::Vector{String},
) where {T<:Tuple}

    sampled_parameters = Tuple(
        i ~ to_submodel(
            prefix(sample_parameters_all_session(prior), parameter_name),
            false,
        ) for (prior, parameter_name) in zip(priors_per_parameter, parameter_names)
    )
    #TODO: avoid type-instabulity when building the Tuple of parameters

    return revert(sampled_parameters)
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
    agent::Agent,
    prior::Dict{String,D},
    data::DataFrame,
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T2},
    grouping_cols::Union{Vector{T3},T3},
    verbose::Bool;
    kwargs...,
) where {
    D<:Distribution,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
}
    #Unless warnings are hidden
    if verbose
        #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
        if any(key -> !(key in keys(prior)), keys(agent.parameters))
            @warn "the agent has parameters which are not estimated. The agent's current parameter values are used as fixed parameters"
        end
    end

    #If there are no parameters to sample
    if length(prior) == 0
        #Throw an error
        throw(ArgumentError("No parameters where specified in the prior."))
    end

    #Check if any keys in the prior occur twice
    if length(keys(prior)) != length(Set(keys(prior)))
        throw(ArgumentError("The prior contains duplicate keys."))
    end
end