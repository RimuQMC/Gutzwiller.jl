"""
    VectorAnsatz(vector::AbstractDVec) <: AbstractAnsatz

A zero-parameter ansatz defined by a state vector.

See also [`AbstractAnsatz`](@ref) and
[`AbstractDVec`](@extref Rimu Rimu.Interfaces.AbstractDVec).
"""
struct VectorAnsatz{A,T,D<:AbstractDVec{A,T}} <: AbstractAnsatz{A,T,0}
    vector::D
end

Rimu.build_basis(va::VectorAnsatz) = collect(keys(va.vector))

(va::VectorAnsatz)(addr, _) = va.vector[addr]
