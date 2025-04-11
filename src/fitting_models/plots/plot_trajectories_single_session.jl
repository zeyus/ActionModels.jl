@recipe function f(
    posterior_states::StateTrajectories,
    session_id::String;
    prior_states::Union{StateTrajectories,Nothing} = nothing,
    states_to_plot::Union{Nothing, String, Vector{String}} = nothing,
)

    throw(ArgumentError("plotting estimated state trajectories for a single session is not yet implemented"))

end




# This should have the full distribution (perhaps boxplots?) of each state
