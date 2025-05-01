Base.@kwdef mutable struct Agent
    action_model::Function
    submodel::Any
    parameters::Dict = Dict()
    initial_state_parameters::Dict{Symbol,InitialStateParameter} = Dict()
    initial_states::Dict{Symbol,InitialStateParameter} = Dict()
    parameter_groups::Dict = Dict()
    states::Dict{Symbol,Any} = Dict(:action => missing)
    history::Dict{Symbol,Vector{Any}} = Dict(:action => [missing])
    save_history::Bool = true
end


