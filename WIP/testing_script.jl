#TODO: Make plot for agent parameters
#TODO: Make parameter_recovery return an axisarray
#TODO: "squeese" density plot
#TODO: make posterior/prior predictive for new data
#TODO: make trajectories into array(per agent) of arraydist (because n_timesteps might vary)
#TODO: rename 'agent' to 'session' where applicable
#TODO: transition to OrderedDicts
#TODO: add progress bar to agent_parameters and trajectories
#TODO: add posterior/prior predictive checks




docs_path = joinpath(@__DIR__, "docs")
using Pkg
Pkg.activate(docs_path)

using Test
using LogExpFunctions

using ActionModels
using Distributions
using DataFrames
using MixedModels
using Turing
import Mooncake


@model function testmodel(bar = [1, 1])

    prior = [arraydist([Normal(), Normal(), Normal()]),
             arraydist([Normal(), Normal()])
             ]

    foo = Vector{Vector{Real}}(undef, length(bar))

    for (idx, barval) in enumerate(bar)
        foo[idx] ~ prior[idx]
    end
end

sample(testmodel(), NUTS(; adtype = AutoReverseDiff(; compile = true)), 100)

sample(testmodel(), NUTS(; adtype = AutoMooncake(; config = nothing)), 100)













function predictive_check(
    agent::Agent,
    agent_parameters::AxisArray,
    inputs::Vector{Array},

)

agent_parameters[]


end







using ActionModels

#Action model which can error 
function action_with_errors(agent, input::R) where {R<:Real}

    noise = agent.parameters["noise"]

    if noise > 2.9
         #Throw an error that will reject samples when fitted
        throw(
            RejectParameters(
                "Rejected noise",
            ),
        )
    end

    actiondist = Normal(input, noise)

    return actiondist
end
#Create agent
new_agent = init_agent(action_with_errors, parameters = Dict("noise" => 1.0))
new_priors = Dict("noise" => truncated(Normal(0.0, 1.5), lower = 0, upper = 4))

#Parameters to be recovered
new_parameter_ranges = Dict(
    "noise" => collect(0:0.5:2.5),
)

#Input sequences to use
input_sequence = [[1, 2, 1, 0, 0, 1, 1, 2, 1, 2], [2, 3, 1, 5, 4, 8, 6, 4, 5]]

#Times to repeat each simulation
n_simulations = 2

#Sampler settings
sampler_settings = (n_iterations = 10, n_chains = 1)

#Run parameter recovery
results_df = parameter_recovery(
    new_agent,
    new_parameter_ranges,
    input_sequence,
    new_priors,
    n_simulations,
    sampler_settings = sampler_settings,
    show_progress = false,
    check_parameter_rejections = true,)







# using ActionModels
# using DataFrames
# using MixedModels
# using Turing
# using StatsPlots



# #1 multiple vectors of same length
# #2 tuples for each formula (formula, prior, link)
# #3 custom Regression types for each formula (inputs: formula, prior, link)
# #4 custom Regression types for each formula (inputs: formula, β, σ link)
# #5 Have all priors as one objects, etc. (maybe custom types) NOPE

# reg_info = 


# model = create_model(
#             agent,
#             [
#                 Regression(
#                     @formula(learning_rate ~ age + (1 | id)),
#                     identity,
#                     RegressionPrior(β = Normal(0, 1), σ = LogNormal()),
#                 ),
#                 Regression(
#                     @formula(action_noise ~ age + (1 | id)),
#                     exp,
#                     RegressionPrior(β = Normal(0, 1),
#                                     σ = [[LogNormal(0, 1), LogNormal(0, 1)], [LogNormal(0, 1)]])
#                 ), 
#             ],
#             data;
#             action_cols = [:actions],
#             input_cols = [:input],
#             grouping_cols = [:id, :treatment],
#         )







# Regression(
#     @formula(),
#     β = [Normal(0, 1), Normal(0, 1)],
#     σ = [Exponential(1), Exponential(1)],
#     inv_link = identity
# )



# ActionModels.RegPrior()

# GG = ActionModels.RegPrior(σ = [TDist(3), TDist(3)])


# plot(TDist(3)*2.5)

# arraydist([Normal(), Normal()]) isa Distributions.Product



# RegPrior()

# RegPrior(
#     TDist(3),
# )

# RegPrior(
#     TDist(3),
#     [TDist(3), TDist(3)],
#     [Exponential(1), Exponential(1)]
# )

# RegPrior(;
#     α = Normal(0, 1),
#     β = [Normal(0, 1), Normal()],
#     σ = [Exponential(1), Exponential(1)]
# )






# Regression(@formula(learning_rate ~1))
# Regression(@formula(learning_rate ~1), RegressionPrior())
# Regression(@formula(learning_rate ~1), identity)
# Regression(@formula(learning_rate ~1), RegressionPrior(), identity)
# Regression(@formula(learning_rate ~1), identity, RegressionPrior())





