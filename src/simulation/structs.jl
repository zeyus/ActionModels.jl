struct Agent{TH<:NamedTuple}
    action_model::Function
    model_attributes::ModelAttributes
    history::TH
    n_timesteps::Int
end

function init_agent(
    action_model::ActionModel;
    save_history::Union{Bool,Symbol, Vector{Symbol}} = false,
)

    ## Initialize model attributes ##
    #Find initial states
    initial_state_parameter_state_names = NamedTuple(
        parameter.state => ParameterDependentState(parameter_name) for
        (parameter_name, parameter) in pairs(action_model.parameters) if
        parameter isa InitialStateParameter
    )
    initial_states = NamedTuple(
        state_name in keys(initial_state_parameter_state_names) ?
        state_name => initial_state_parameter_state_names[state_name] :
        state_name => state.initial_value for
        (state_name, state) in pairs(action_model.states)
    )

    #Initialize model attributes
    model_attributes = initialize_attributes(action_model, initial_states)

    #Reset it, so that the initial states are set to the initial values
    reset!(model_attributes)

    ## Initialize history ##
    #Find states for which to save history
    if save_history isa Bool
        #If save_history is true, save all states
        if save_history
            save_history = collect(keys(model_attributes.states))
        else
            #If save_history is false, don't save any states
            save_history = Symbol[]
        end
    end
    if save_history isa Symbol
        #If save_history is a symbol, save only that state
        save_history = [save_history]
    end

    #Initialize history with the initial states
    history = NamedTuple(
        state_name => push!(
            Vector{Union{Missing,action_model.states[state_name].type}}(),
            return_value(model_attributes.initial_states[state_name]),
        ) for state_name in save_history
    )

    ## Create agent ##
    Agent(action_model.action_model, model_attributes, history, Variable(0))
end

#Helper function for dealing with initial states and other places that mix Variables and fixed values
function return_value(variable::Variable)
    return variable.value
end
function return_value(value::T) where {T}
    return value
end