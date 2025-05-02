#######################################################################
### FUNCTION FOR RESHAPING A VECTOR OF TUPLES TO A TUPLE OF VECTORS ###
#######################################################################
function evert(v::Vector{<:NTuple{N,Any}}) where {N}
    n = length(v)

    # Create result vectors with the correct types
    sample = first(v)
    result = ntuple(i -> Vector{typeof(sample[i])}(undef, n), N)

    # Fill each vector with values from the corresponding tuple position
    @inbounds for i = 1:n
        t = v[i]
        for j = 1:N
            result[j][i] = t[j]
        end
    end

    return result
end

### DISPATCH FOR THE CASE WHEN IT IS REALS OR DISTS INSTEAD OF TUPLE ###
function evert(v::Vector{T}) where {T}
    return (v,)
end



#####################################################################################################
####### FUNCTIONS FOR EXTRACTING A VALUE WHICH WORKS WITH DIFFERENT AUTODIFFERENTIATION BACKENDS ####
#####################################################################################################
function ad_val(x::ReverseDiff.TrackedReal)
    return ReverseDiff.value(x)
end
function ad_val(x::ReverseDiff.TrackedArray)
    return ReverseDiff.value(x)
end
function ad_val(x::ForwardDiff.Dual)
    return ForwardDiff.value(x)
end
function ad_val(x::Real)
    return x
end




############################################
### BOUNDED VERSION OF EXP AND LOGISTIC ####
############################################
function bounded_exp(; lower = nothing, upper = 1e200)

    return function _bounded_exp(
        x::T;
        lower = if isnothing(lower)
            eps(T)
        else
            lower
        end,
        upper = upper,
    ) where {T<:Real}

        return clamp(exp(x), lower, upper)
    end
end

function bounded_logistic(; lower = nothing, upper = nothing)

    return function _bounded_logistic(
        x::T;
        lower = if isnothing(lower)
            eps(T)
        else
            lower
        end,
        upper = if isnothing(upper)
            1-eps(T)
        else
            upper
        end,
    ) where {T<:Real}

        return clamp(logistic(x), lower, upper)
    end
end
