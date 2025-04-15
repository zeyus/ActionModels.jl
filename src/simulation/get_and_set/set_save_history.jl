#Function for changing the save_history setting
function set_save_history!(agent::Agent, save_history::Bool)

    #Change it in the agent
    agent.save_history = save_history

    #And in its submodel
    set_save_history!(agent.submodel, save_history)
end

#If there is an empty submodel, do nothing
function set_save_history!(submodel::Nothing, save_history::Bool)
    return nothing
end
