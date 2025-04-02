# Benchmark Nested Comprehension vs. For Loop
# Run this code in a Julia environment with BenchmarkTools installed

using BenchmarkTools

# Define a simple struct for testing
struct MockDistribution
    value::Float64
end

# Mock functions to simulate the agent operations
function set_params!(state, params)
    return state + sum(params)
end

function reset_state!(state)
    return 0.0
end

function action_model(state, input)
    return MockDistribution(input * state)
end

function update_state!(state, action)
    return state + action
end

# Generate test data
function make_test_data(n_sessions, n_steps)
    params = [rand(3) for _ in 1:n_sessions]
    inputs = [[rand() for _ in 1:n_steps] for _ in 1:n_sessions]
    actions = [[rand() for _ in 1:n_steps] for _ in 1:n_sessions]
    return params, inputs, actions
end

# Approach using nested list comprehensions
function nested_comprehension(params, inputs, actions)
    state = 1.0
    distributions = [
        begin
            state = set_params!(state, session_params)
            state = reset_state!(state)
            [
                begin
                    dist = action_model(state, input)
                    state = update_state!(state, action)
                    dist
                end for (input, action) in zip(session_inputs, session_actions)
            ]
        end for (session_params, session_inputs, session_actions) in
        zip(params, inputs, actions)
    ]
    return distributions
end

# Approach using for loops with pre-allocation
function for_loop(params, inputs, actions)
    n_sessions = length(params)
    distributions = Vector{Vector{MockDistribution}}(undef, n_sessions)
    
    state = 1.0
    for i in 1:n_sessions
        state = set_params!(state, params[i])
        state = reset_state!(state)
        
        n_steps = length(inputs[i])
        session_dists = Vector{MockDistribution}(undef, n_steps)
        
        for j in 1:n_steps
            dist = action_model(state, inputs[i][j])
            state = update_state!(state, actions[i][j])
            session_dists[j] = dist
        end
        
        distributions[i] = session_dists
    end
    
    return distributions
end

# Run the benchmark
n_sessions = 100
n_steps = 50
params, inputs, actions = make_test_data(n_sessions, n_steps)

# Verify correctness
list_result = nested_comprehension(params, inputs, actions)
loop_result = for_loop(params, inputs, actions)

# Run benchmarks
println("Benchmarking List Comprehension:")
list_bench = @benchmark nested_comprehension($params, $inputs, $actions)
display(list_bench)

println("\nBenchmarking For Loop:")
loop_bench = @benchmark for_loop($params, $inputs, $actions)
display(loop_bench)

# Compare results
println("\nTime comparison:")
list_median = median(list_bench.times) / 1e6  # convert to milliseconds
loop_median = median(loop_bench.times) / 1e6
println("List comprehension median time: ", list_median, " ms")
println("For loop median time: ", loop_median, " ms")
println("Speedup: ", list_median / loop_median, "x")

println("\nMemory comparison:")
println("List comprehension allocations: ", list_bench.allocs)
println("For loop allocations: ", loop_bench.allocs)
println("Memory used by list comprehension: ", list_bench.memory, " bytes")
println("Memory used by for loop: ", loop_bench.memory, " bytes")
println("Memory reduction ratio: ", list_bench.memory / loop_bench.memory, "x")
