@recipe function f(
    posterior_states::StateTrajectories,
    session_id::String;
    prior_states = nothing, #::Union{StateTrajectories,Nothing}
    states_to_plot = nothing, #::Union{Nothing, String, Vector{String}}
)

    throw(ArgumentError("plotting estimated state trajectories for a single session is not yet implemented"))

end




# This should have the full distribution (perhaps boxplots?) of each state
