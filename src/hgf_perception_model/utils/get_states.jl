"""
"""
function get_states(hgf::HGFStruct, target_state::String)

    # Split state name to get node and state name
    (node_name, state_name) = split(target_state, "__", limit = 2)

    #Check that the node exists
    if node_name in keys(hgf.all_nodes)
        #If th eprediction mean has been specified
        if state_name == "prediction_mean"
            #Get the node
            node = hgf.all_nodes[node_name]
            #Get the new prediction mean
            state = calculate_prediction_mean(node, node.value_parents)
        #If another prediction state has been specified
        elseif state_name in [
            "prediction_volatility",
            "prediction_precision",
            "auxiliary_prediction_precision",
        ]
            #Get the new prediction
            prediction = get_prediction(node)
            #And get the specified state from it
            state = getproperty(prediction, Symbol(state_name))
        else
            #Get the state from the node
            state = getproperty(hgf.all_nodes[node_name].state, Symbol(state_name))
        end
    else
        #If it doesn't exist, throw an error
        error("The node " * node_name * " does not exist")
    end

    return state
end

"""
"""
function get_states(hgf::HGFStruct, target_states::Array{String})
    #Initialize tuple for storing states
    state_list = (;)

    #Go through each state
    for state in target_states
        #Add its state to the tuple
        state_list = merge(state_list, (Symbol(state) => get_states(hgf, state),))
    end

    return state_list
end

"""
"""
function get_states(hgf::HGFStruct)

    #Initialize list for target states
    target_states = String[]

    #Go through each node
    for node in hgf.ordered_nodes.all_nodes
        #Go through each state in the node
        for state_name in fieldnames(typeof(node.state))
            #Add the name to the list of target states
            push!(target_states, node.name * "__" * String(state_name))
        end
    end

    #Get the states
    state_list = get_history(hgf, target_states)

    return state_list
end