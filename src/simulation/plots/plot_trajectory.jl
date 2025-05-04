@recipe function f(
    agent::Agent,
    target_state::Symbol;
)
    #Get the history of the state
    state_history = get_history(agent, target_state)
    #Replace missings with NaNs for plotting
    state_history = replace(state_history, missing => NaN)

    #The x-axis starts at 0
    x_axis = collect(0:length(state_history)-1)

    xlabel --> "timestep"
    yguide --> "$target_state"

    #Plot the history
    @series begin
        seriestype --> :path
        markersize --> 5
        title --> "$target_state"
        x_axis, state_history
    end
end