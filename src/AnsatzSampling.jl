using Rimu.Hamiltonians: ModifiedHamiltonian, TransformUndoer, parent_operator

struct AnsatzSampling{Adj,T,N,A<:AbstractAnsatz{<:Any,T,N},H} <: ModifiedHamiltonian{T}
    hamiltonian::H
    ansatz::A
    params::SVector{N,T}

    function AnsatzSampling{Adj}(
        hamiltonian::H, ansatz::A, params::SVector{N,T}
    ) where {Adj,T,N,A<:AbstractAnsatz{<:Any,T,N},H}
        return new{Adj,T,N,A,H}(hamiltonian, ansatz, params)
    end
end

function AnsatzSampling(
    h, ansatz::AbstractAnsatz{<:Any,<:Any,N}, params::Vararg{Number,N}
) where {N}

    return AnsatzSampling(h, ansatz, params)
end

function AnsatzSampling(h, ansatz::AbstractAnsatz{K,T,N}, params) where {K,T,N}
    # sanity checks
    if typeof(starting_address(h)) ≢ K
        throw(ArgumentError("Ansatz keytype does not match Hamiltonian starting_address"))
    end
    if T ≢ promote_type(T, eltype(h))
        throw(ArgumentError("Hamiltonian and Ansatz eltypes don't match"))
    end

    params = SVector{N,T}(params)

    return AnsatzSampling{false}(h, ansatz, params)
end

Rimu.Hamiltonians.parent_operator(h::AnsatzSampling) = h.hamiltonian
Rimu.Hamiltonians.modify_diagonal(h::AnsatzSampling, _, value) = value

Rimu.LOStructure(h::AnsatzSampling) = LOStructure(parent_operator(h))
function Rimu.adjoint(h::AnsatzSampling{A}) where {A}
    return AnsatzSampling{!A}(h.hamiltonian', h.ansatz, h.params)
end

function Rimu.Hamiltonians.modify_offdiagonal(h::AnsatzSampling{A}, src, dst, value) where {A}
    src_ansatz = (h.ansatz)(src, h.params)
    dst_ansatz = (h.ansatz)(dst, h.params)
    if !A
        return dst => value * (dst_ansatz / src_ansatz)
    else
        return dst => value * (src_ansatz / dst_ansatz)
    end
end

###
### TransformUndoer
###
const AnsatzTransformUndoer{O} = TransformUndoer{<:Any,<:AnsatzSampling,O}

function TransformUndoer(k::AnsatzSampling, op::AbstractOperator)
    T = promote_type(eltype(k) * eltype(op))
    return Rimu.Hamiltonians.TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

# methods for general operator `f^{-1} A f^{-1}`
Rimu.LOStructure(::Type{<:AnsatzTransformUndoer{A}}) where {A} = LOStructure(A)

function Rimu.Hamiltonians.modify_diagonal(s::AnsatzTransformUndoer, addr, value)
    return value / s.transform.ansatz(addr, s.transform.params)^2
end
function Rimu.Hamiltonians.modify_offdiagonal(s::AnsatzTransformUndoer, src, dst, value)
    ansatz = s.transform.ansatz
    params = s.transform.params

    ansatz1 = ansatz(src, params)
    ansatz2 = ansatz(dst, params)
    return dst, value / (ansatz1 * ansatz2)
end
