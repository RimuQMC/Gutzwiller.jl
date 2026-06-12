

"""
    CoherentAnsatz
Ansatz for a coherent state wavefunction.
```math
Ψ(n_k) = ∏_k e^{\\frac{-|α_k^2|}{2} \\frac{α_k^{n_k}}{√{n_k!}}
```
where α_k are variational parameters.
"""

struct CoherentAnsatz{A,T,M,H} <: AbstractAnsatz{A,Float64,M}
    hamiltonian::H
    fact_table::Vector{Float64}
end

function CoherentAnsatz(hamiltonian)
    addr_type = typeof(starting_address(hamiltonian))
    M = num_modes(starting_address(hamiltonian))

    mode_cutoff = num_particles(starting_address(hamiltonian))
    if ismissing(mode_cutoff)            
        mode_cutoff = isnothing(hamiltonian.mode_cutoff) ? 255 : hamiltonian.mode_cutoff
    end
    fact_table = [log(sqrt(factorial(big(n)))) for n in 0:mode_cutoff]
    return CoherentAnsatz{addr_type,Float64,M,typeof(hamiltonian)}(hamiltonian,fact_table)
end

Rimu.build_basis(ca::CoherentAnsatz) = build_basis(ca.hamiltonian)

function (ca::CoherentAnsatz)(addr, params)
    occ = onr(addr)

    if !isnothing(ca.hamiltonian.mode_cutoff) &&
       any(x -> x > ca.hamiltonian.mode_cutoff, occ)
        return 0.0
    end
    
    logval = 0.0
    @inbounds for (i, ni) in enumerate(occ)
        if ni != 0
            logval += ni*log(params[i]) - ca.fact_table[ni+1]
        end
    end
    normterm = 0.0
    @inbounds for i in eachindex(params)
        normterm += params[i]^2
    end
    logval -= 0.5 * normterm 
    val = exp(logval)

    return val
end

function val_and_grad(ca::CoherentAnsatz, addr, params)
    val = ca(addr, params)
    grad = SVector{length(occ),Float64}(
        (occ[i]/params[i] -  params[i]) * val for i in eachindex(occ)
    )
    return val, grad
end
