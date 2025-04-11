@recipe function f(
    posterior_parameters::SessionParameters;
    prior_parameters::Union{SessionParameters,Nothing} = nothing,
    parameters_to_plot::Union{Nothing, String, Vector{String}} = nothing,
)

    throw(ArgumentError("plotting estimated parameters for all sessions is not yet implemented"))

end



## This should be all the session's parameters above each other, like the stan_plot / PlotInd of HBayesDM, with the priors in grey behind