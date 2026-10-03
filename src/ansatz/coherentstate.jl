"""
    CoherentAnsatz(hamiltonian) <: AbstractAnsatz

Ansatz for a coherent state wave function.
```math
C(|n_1, n_2, n_3, …⟩; \\mathbf{α}) = ∏_{k=1}^M \\frac{α_k^{n_k}}{√{n_k!}} e^{\\frac{-|α_k^2|}{2}}
```
where α_k are variational parameters.
"""
struct CoherentAnsatz{A,H,M} <: AbstractAnsatz{A,Float64,M}
    hamiltonian::H
end
function CoherentAnsatz(hamiltonian::H) where {H}
    A = starting_address(hamiltonian)
    M = num_modes_check_equal(A)
    return CoherentAnsatz{A,H,M}(hamiltonian)
end

Rimu.build_basis(ca::CoherentAnsatz) = build_basis(ca.hamiltonian)

@inline function (ca::CoherentAnsatz)(addr::SingleComponentFockAddress, params)
    @boundscheck begin
        if length(params) != num_parameters(ca)
            throw(ArgumentError("`CoherentAnsatz` with $(num_parameters(ca)) parameters called with $(length(params)) parameters"))
        elseif length(params) != num_modes(addr)
            throw(ArgumentError("`CoherentAnsatz` with $(num_parameters(ca)) parameters called with an address with $(num_modes(addr)) modes"))
        end
    end
    logval = 0.0
    @inbounds for (occnum, mode) in occupied_modes(addr)
        occnum > maximum_mode_occupation(ca.hamiltonian) && return 0.0
        logval += occnum * log(params[mode]) - loggamma(occnum + 1)/2
    end
    logval -= sum(abs2, params) / 2
    return exp(logval)
end

function val_and_grad(ca::CoherentAnsatz, addr, params)
    val = ca(addr, params)
    occ = onr(addr)
    grad = SVector{length(occ),Float64}(
        (occ[i] / params[i] - params[i]) * val for i in eachindex(occ)
    )
    return val, grad
end
