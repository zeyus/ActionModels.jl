### FUNCTION FOR RESHAPING A VECTOR OF TUPLES TO A TUPLE OF VECTORS ###
function evert(v::Vector{<:NTuple{N,Any}}) where N
    n = length(v)
    
    # Create result vectors with the correct types
    sample = first(v)
    result = ntuple(i -> Vector{typeof(sample[i])}(undef, n), N)
    
    # Fill each vector with values from the corresponding tuple position
    @inbounds for i in 1:n
        t = v[i]
        for j in 1:N
            result[j][i] = t[j]
        end
    end
    
    return result
end

### DISPATCH FOR THE CASE WHEN IT IS REALS OR DISTS INSTEAD OF TUPLE ###
function evert(v::Vector{T}) where T<:Union{<:Distribution, <:Real, <:Union{<:Real, Missing}}
    return (v,)
end



### FUNCTION FOR CONVERTING A TUPLE OF VECTORS INTO A VECTOR OF TUPLES ###
function revert(t::NTuple{N,Vector{T}}) where {N, T}
    n = length(first(t))                     # Length of the vectors
    result = Vector{NTuple{N, T}}(undef, n)  # Preallocate result vector

    @inbounds for i in 1:n
        result[i] = ntuple(j -> t[j][i], N)  # Create a tuple for each index
    end

    return result
end