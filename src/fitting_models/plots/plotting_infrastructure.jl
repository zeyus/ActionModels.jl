#Plotting a ModelFit just plots the session parameters
@recipe function f(modelfit::ModelFit; kwargs...)

    #Get session parameters
    posterior_parameters = get_session_parameters!(modelfit, :posterior)
    prior_parameters = get_session_parameters!(modelfit, :prior)

    plot(posterior_parameters; prior_parameters = prior_parameters, kwargs...)

end

### Function for plotting session parameters ###
function plot_session_parameters(
    modelfit::ModelFit;
    session_id::Union{Nothing,String} = nothing,
    parameters_to_plot::Union{Nothing,String,Vector{String}} = nothing,
    plot_prior::Bool = true,
    kwargs...,
)
    #Get session parameters
    posterior_parameters = get_session_parameters!(modelfit, :posterior)
    if plot_prior == true
        prior_parameters = get_session_parameters!(modelfit, :prior)
    else
        prior_parameters = nothing
    end

    if isnothing(session_id)
        plot(
            posterior_parameters;
            prior_parameters = prior_parameters,
            parameters_to_plot = parameters_to_plot,
            kwargs...,
        )
    else
        plot(
            posterior_parameters,
            session_id;
            prior_parameters = prior_parameters,
            parameters_to_plot = parameters_to_plot,
            kwargs...,
        )
    end
end

### Function for plotting state trajectories ###
function plot_state_trajectories(
    modelfit::ModelFit;
    session_id::Union{Nothing,String} = nothing,
    states_to_plot::Union{Nothing,String,Vector{String}} = nothing,
    plot_prior::Bool = true,
    kwargs...,
)
    #Get state trajectories
    posterior_states = get_state_trajectories!(modelfit, :posterior)
    if plot_prior == true
        prior_states = get_state_trajectories!(modelfit, :prior)
    else
        prior_states = nothing
    end

    if isnothing(session_id)
        plot(
            posterior_states;
            prior_states = prior_states,
            states_to_plot = states_to_plot,
            kwargs...,
        )
    else
        plot(
            posterior_states,
            session_id;
            prior_states = prior_states,
            states_to_plot = states_to_plot,
            kwargs...,
        )
    end
end