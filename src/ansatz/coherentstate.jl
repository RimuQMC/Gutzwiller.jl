"""
    CoherentAnsatz(hamiltonian) <: AbstractAnsatz

Ansatz for a coherent state wave function.
```math
C(|n_1, n_2, n_3, …⟩; \\mathbf{α}) = ∏_{k=1}^M \\frac{α_k^{n_k}}{\\sqrt{n_k!}} \\exp(\\frac{-|α_k^2|}{2})
```
where ``α_k`` are `M` variational parameters and `M` is the number of modes.

The ansatz only supports Hamiltonians with
[`BoseFS`](@extref Rimu Rimu.BitStringAddresses.BoseFS) addresses.

See also [`AbstractAnsatz`](@ref).
"""
struct CoherentAnsatz{A<:BoseFS,H,M} <: AbstractAnsatz{A,Float64,M}
    hamiltonian::H
end
function CoherentAnsatz(hamiltonian::H) where {H}
    A = typeof(starting_address(hamiltonian))
    if !(A <: BoseFS)
        throw(ArgumentError("only `BoseFS` addresses are supported."))
    end
    M = num_modes_check_equal(A)
    return CoherentAnsatz{A,H,M}(hamiltonian)
end

Rimu.build_basis(ca::CoherentAnsatz) = build_basis(ca.hamiltonian)

function (ca::CoherentAnsatz)(addr::BoseFS, params)
    logval = 0.0
    for (occnum, mode) in occupied_modes(addr)
        logval += occnum * log(params[mode]) - loggamma(occnum + 1)/2
    end
    logval -= sum(abs2, params) / 2
    return exp(logval)
end

function val_and_grad(ca::CoherentAnsatz, addr::BoseFS, params)
    val = ca(addr, params)
    occ = onr(addr)
    grad = SVector{length(occ),Float64}(
        (occ[i] / params[i] - params[i]) * val for i in eachindex(occ)
    )
    return val, grad
end
