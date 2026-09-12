"""
    CombinationAnsatz

Superposition of `ansatz1` and `ansatz2`. Has one more parameter than each ansatz controling
the mixing between them.

See also [`AbstractAnsatz`](@ref).
"""
struct CombinationAnsatz{
    K,V,N,A<:AbstractAnsatz{K,V},B<:AbstractAnsatz{K,V}
} <: AbstractAnsatz{K,V,N}
    ansatz1::A
    ansatz2::B
end

function Base.:+(a1::AbstractAnsatz{K,V,N}, a2::AbstractAnsatz{K,V,M}) where {K,V,N,M}
    return CombinationAnsatz{K,V,N + M + 1,typeof(a1),typeof(a2)}(a1, a2)
end
@generated function (ca::CombinationAnsatz{K,V,<:Any,A,B})(addr::K, params) where {K,V,A,B}
    # Generated to ensure params are handled without allocations
    N = num_parameters(A)
    M = num_parameters(B)
    range1 = SVector{N,Int}(1:N)
    range2 = SVector{M,Int}((N+1):(N+M))
    quote
        p1 = params[$range1]
        p2 = params[$range2]
        α = params[end]

        return α * ca.ansatz1(addr, p1) + (1 - α) * ca.ansatz2(addr, p2)
    end
end

@generated function val_and_grad(
    ca::CombinationAnsatz{K,V,NP,A,B}, addr::K, params
) where {K,V,NP,A,B}
    # Generated to ensure params are handled without allocations
    N = num_parameters(A)
    M = num_parameters(B)
    range1 = SVector{N,Int}(1:N)
    range2 = SVector{M,Int}((N+1):(N+M))
    quote
        p1 = params[$range1]
        p2 = params[$range2]
        α = params[end]
        val1, grad1 = val_and_grad(ca.ansatz1, addr, p1)
        val2, grad2 = val_and_grad(ca.ansatz2, addr, p2)
        grad1 *= α
        grad2 *= (1 - α)

        return α * val1 + (1 - α) * val2, SVector{NP,V}(grad1..., grad2..., val1 - val2)
    end
end
