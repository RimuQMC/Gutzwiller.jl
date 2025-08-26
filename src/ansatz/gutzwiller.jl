"""
    GutzwillerAnsatz(hamiltonian) <: AbstractAnsatz

The Gutzwiller ansatz:

```math
G(|f⟩; g) = exp(-g ⟨f|H|f⟩),
```

where ``H`` is the `hamiltonian` passed to the struct.

It takes a single parameter, `g`.
"""
struct GutzwillerAnsatz{A,T<:Real,H} <: AbstractAnsatz{A,T,1}
    hamiltonian::H
end
function GutzwillerAnsatz(hamiltonian)
    A = typeof(starting_address(hamiltonian))
    T = eltype(hamiltonian)
    return GutzwillerAnsatz{A,T,typeof(hamiltonian)}(hamiltonian)
end

Rimu.build_basis(gv::GutzwillerAnsatz) = build_basis(gv.hamiltonian)

function val_and_grad(gv::GutzwillerAnsatz, addr, params)
    g = only(params)
    diag = diagonal_element(gv.hamiltonian * addr)

    val = exp(-g * diag)
    der = -diag * val

    return val, SVector(der)
end
function (gv::GutzwillerAnsatz)(addr, params)
    return exp(-only(params) * diagonal_element(gv.hamiltonian * addr))
end
