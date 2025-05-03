# function Base.show(io::IO, ::MIME"text/plain", agent::Agent)

#     ## Get information from agent struct
#     action_model_name = string(agent.action_model)
#     n_observations = length(agent.history[:action]) - 1

#     ## Build the output string
#     output = IOBuffer()
#     println(output, "-- ActionModels Agent --")
#     println(output, "Action model: $action_model_name")

#     # Number of observations
#     println(output, "This agent has received $n_observations observations")


#     ## Print the final string
#     print(io, String(take!(output)))
# end
