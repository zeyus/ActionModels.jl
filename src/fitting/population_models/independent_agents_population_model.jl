############################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A SINGLE-AGENT PRIOR ###
############################################################################################################

struct IndependentPopulationModel <: AbstractPopulationModel end

function create_model(
    agent::Agent,
    prior::Dict{String,D},
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T2},
    grouping_cols::Union{Vector{T3},T3} = Vector{String}(),
    kwargs...,
) where {
    D<:Distribution,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
}

    #Get number of sessions
    n_sessions = length(groupby(data, grouping_cols))

    #Get the names of the estimated parameters
    parameter_names = collect(keys(prior))

    #Create a filldist for each parameter
    priors_per_parameter = Tuple([filldist(prior[parameter_name], n_sessions) for parameter_name in parameter_names])

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
        kwargs...,
    )
end




#######################################################################################################
### SIMPLE STATISTICAL MODEL WHERE AGENTS ARE INDEPENDENT AND THEIR PARAMETERS HAVE THE SAME PRIORS ###
#######################################################################################################
@model function independent_population_model(
    priors_per_parameter::T,
    parameter_names::Vector{String},
) where {T<:Tuple}




    return 
end












##########################################################################
####### FUNCTION FOR RENAMING CHAINS FOR A SIMPLE STATISTICAL MODEL ######
##########################################################################
function rename_chains(
    chains::Chains,
    model::DynamicPPL.Model,
    #Arguments from statistical model
    prior::Dict{T,D},
    n_agents::I,
    agent_parameters::Vector{Dict{Any,Real}},
) where {T<:Union{String,Tuple,Any},D<:Distribution,I<:Int}

    #Extract agent ids
    agent_ids = model.args.agent_ids

    ## Make dict with replacement names ##
    replacement_names = Dict{String,String}()

    for (agent_idx, agent_id) in enumerate(agent_ids)

        #Go through each parameter in the prior
        for (parameter_idx, parameter_key) in enumerate(keys(prior))

            #If the parameter key is a tuple
            if parameter_key isa Tuple
                #Join the tuple with double underscores
                parameter_key_right = join(parameter_key, tuple_separator)
            else
                #Otherwise, keep it as it is
                parameter_key_right = parameter_key
            end

            #Set a replacement name
            replacement_names["parameters[$parameter_idx, $agent_idx]"] = "$(agent_id).$parameter_key_right"
        end
    end

    #Replace names in the fitted model and return it
    replacenames(chains, replacement_names)
end


#################################################################
####### CHECKS TO BE MADE FOR THE SIMPLE STATISTICAL MODEL ######
#################################################################
function check_population_model(
    #Arguments from statistical model
    prior::Dict{T,D},
    n_agents::I,
    agent_parameters::Vector{Dict{Any,Real}};
    #Arguments from check_model
    verbose::Bool = true,
    agent::Agent,
) where {T<:Union{String,Tuple,Any},D<:Distribution,I<:Int}
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
end
