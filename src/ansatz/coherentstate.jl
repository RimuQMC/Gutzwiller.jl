
"""
CoherentAnsatz
Ansatz for a coherent state wavefunction.
```math
\\Psi{n_k} = \\prod_{k} e^{\\frac{-\\abs{\\alpha_{k}}^2} }{2} \\frac{\\alpha_{k}^{n_k}}{\\sqrt{n_{k}!}}
```
where ``\\alpha_{k}`` are variational parameters.
"""

struct CoherentAnsatz{A,T,M,H} <: AbstractAnsatz{A,Float64,M}
    hamiltonian::H
    fact_table::Vector{Float64}
end



function CoherentAnsatz(hamiltonian)
    addr_type = typeof(starting_address(hamiltonian))
    M = num_modes(starting_address(hamiltonian))

    mode_cutoff = hamiltonian.mode_cutoff === nothing ? 255 : hamiltonian.mode_cutoff
    fact_table = [sqrt(factorial(big(n))) for n in 0:mode_cutoff]
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
            ai = exp(params[i])
            logval += ni*log(ai) - log(ca.fact_table[ni+1])
        end
    end
    normterm = 0.0
    @inbounds for i in eachindex(params)
        ai = exp(params[i])
        normterm += ai^2
    end
    logval -= 0.5  * normterm 
    val = exp(logval)

    return val
end

function val_and_grad(ca::CoherentAnsatz, addr, params)
    occ = onr(addr)

    if !isnothing(ca.hamiltonian.mode_cutoff) &&
       any(x -> x > ca.hamiltonian.mode_cutoff, occ)
        return 0.0, SVector{length(occ),Float64}(zeros(length(occ)))
    end

    
    logval = 0.0
    @inbounds for (i, ni) in enumerate(occ)
        if ni != 0
            ai = exp(params[i])
            logval += ni*log(ai) - log(ca.fact_table[ni+1])
        end
    end
    normterm = 0.0
    @inbounds for i in eachindex(params)
        ai = exp(params[i])
        normterm += ai^2
    end
    logval -= 0.5  * normterm 
    val = exp(logval)

    grad = SVector{length(occ),Float64}(
        (occ[i] - exp(2 * params[i])) * val for i in eachindex(occ)
    )

    return val, grad
end
