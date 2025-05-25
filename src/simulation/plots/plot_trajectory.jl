@recipe function f(
    agent::Agent,
    target_state::Symbol,
    index::Union{Nothing,Vector{Int}} = nothing;
)
    #Get the history of the state
    state_history = get_history(agent, target_state)
    
    if !isnothing(index) && !(first(state_history) isa AbstractArray)
        @error "And index was provided, but the state is not an array."
    end

    if isnothing(index) && first(state_history) isa AbstractArray
        @error "The state is an array. Provide an index to select which value to plot."
    end
    
    #If the state is multivariate
    if first(state_history) isa AbstractArray && !isnothing(index)
        
        #Extract the state history for the given index
        index = CartesianIndex(index...)
        state_history = map(
            i -> state_history[index],
            index,
        )
    end

    #Replace missings with NaNs for plotting
    state_history = replace(state_history, missing => NaN)

    #The x-axis starts at 0
    x_axis = collect(0:(length(state_history)-1))

    xlabel --> "Timestep"
    yguide --> "$target_state"

    #Plot the history
    @series begin
        label --> nothing
        seriestype --> :path
        markersize --> 5
        title --> "$target_state"
        x_axis, state_history
    end
end