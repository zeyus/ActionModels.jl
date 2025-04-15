# function pvl_delta(agent::Agent, input::Tuple{Int64,Float64})

#     deck, reward = input

#     learning_rate = agent.parameters["learning_rate"]
#     reward_sensitivity = agent.parameters["reward_sensitivity"]
#     loss_aversion = agent.parameters["loss_aversion"]
#     inv_temperature = agent.parameters["inv_temperature"]

#     expected_value = agent.states["expected_value"]

#     #Get action probabilities by softmaxing expected values for each deck
#     action_probabilities = softmax(expected_value * inv_temperature)

#     #Avoid underflow and overflow
#     action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
#     action_probabilities = action_probabilities / sum(action_probabilities)

#     #Calculate prediction error
#     if reward >= 0
#         prediction_error = (reward^reward_sensitivity) - expected_value[deck]
#     else
#         prediction_error =
#             -loss_aversion * (abs(reward)^reward_sensitivity) - expected_value[deck]
#     end

#     #Update expected values
#     new_expected_value = [
#         expected_value[deck_idx] + learning_rate * prediction_error * (deck == deck_idx) for deck_idx = 1:4
#     ]

#     update_states!(agent, "expected_value", new_expected_value)

#     return Categorical(action_probabilities)
# end

# # agent = init_agent(
# #     pvl_delta,
# #     parameters = Dict(
# #         "learning_rate" => 0.1,
# #         "reward_sensitivity" => 0.5,
# #         "inv_temperature" => 1,
# #         "loss_aversion" => 1,
# #     ),
# #     states = Dict("expected_value" => zeros(Float64, 4)),
# # )