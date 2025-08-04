"""
    LocalEnergyEvaluator(hamiltonian, ansatz; basis=build_basis(hamiltonian))

Wrapper over `hamiltonian` and `ansatz` that can be used to efficiently compute local
energies. Stores the basis, which can be passed throught the `basis` keyword argument.

Instances of `LocalEnergyEvaluator` can be called with parameters to return a local energy.
Alternatively, `val_and_grad` can be used to compute the value and gradient (with respect to
the parameters).

```jldoctest
julia> using Rimu, Gutzwiller

julia> H = HubbardReal1D(BoseFS((1,1,1,1,1)));

julia> le = LocalEnergyEvaluator(H, GutzwillerAnsatz(H));

julia> le(0.5)
-8.22150794790878

julia> val_and_grad(le, 0.5)
(-8.22150794790878, [0.2718589892870141])
```

# Extended help

## Using with Optim.jl

You can use the gradient provided by this struct by wrapping it in `Optim.only_fg!`.

```jldoctest
julia> using Rimu, Gutzwiller

julia> H = HubbardReal1D(BoseFS((1,1,1,1,1)));

julia> le = LocalEnergyEvaluator(H, GutzwillerAnsatz(H));

julia> optimize(le, [0.5])
 * Status: success

 * Candidate solution
    Final objective value:     -8.225604e+00

 * Found with
    Algorithm:     Nelder-Mead

 * Convergence measures
    √(Σ(yᵢ-ȳ)²)/n ≤ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    9
    f(x) calls:    21


julia> optimize(Optim.only_fg!(le), [0.5])
 * Status: success

 * Candidate solution
    Final objective value:     -8.225604e+00

 * Found with
    Algorithm:     L-BFGS

 * Convergence measures
    |x - x'|               = 1.18e-05 ≰ 0.0e+00
    |x - x'|/|x'|          = 2.51e-05 ≰ 0.0e+00
    |f(x) - f(x')|         = 6.72e-10 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 8.17e-11 ≰ 0.0e+00
    |g(x)|                 = 4.54e-11 ≤ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    3
    f(x) calls:    7
    ∇f(x) calls:   7


```
"""
struct LocalEnergyEvaluator{N,T,H,A,D<:AbstractAnsatz{A,T,N}}
    hamiltonian::H
    ansatz::D
    basis::Vector{A}
end
function LocalEnergyEvaluator(ham, ansatz; basis=build_basis(ham))
    return LocalEnergyEvaluator(ham, ansatz, basis)
end

function Base.show(io::IO, le::LocalEnergyEvaluator)
    print(io, "LocalEnergyEvaluator(", le.hamiltonian, ", ", le.ansatz, ")")
end

function val_and_grad(le::LocalEnergyEvaluator{N,T}, params) where {N,T}
    return val_and_grad(le, SVector{N,T}(params))
end

function val_and_grad(le::LocalEnergyEvaluator{N,T}, params::SVector{N,T}) where {N,T<:Real}
    # numerator, denominator and their gradients
    init = (zero(T), zero(T), zero(SVector{N,T}), zero(SVector{N,T}))

    result = Folds.mapreduce(add, le.basis; init) do k
        ham = le.hamiltonian

        k_val, k_grad = val_and_grad(le.ansatz, k, params)
        column = ham * k
        diag = diagonal_element(column)

        num = abs2(k_val) * diag
        num_grad = 2 * k_grad * k_val * diag

        den = abs2(k_val)
        den_grad = 2 * k_val * k_grad

        for (l, melem) in offdiagonals(column)
            l_val, l_grad = val_and_grad(le.ansatz, l, params)
            num += l_val * melem * k_val
            num_grad += melem * (l_val * k_grad + l_grad * k_val)
        end

        (num, den, num_grad, den_grad)
    end

    numerator, denominator, numerator_grad, denominator_grad = result

    return (
        numerator / denominator,
        (numerator_grad * denominator - numerator * denominator_grad) / denominator^2
    )
end

function (le::LocalEnergyEvaluator{N,T})(params) where {N,T}
    # numerator, denominator
    init = (zero(T), zero(T))

    result = Folds.mapreduce(add, le.basis; init) do k
        ham = le.hamiltonian

        k_val = le.ansatz(k, params)
        column = ham * k
        diag = diagonal_element(column)

        num = abs2(k_val) * diag
        den = abs2(k_val)

        for (l, melem) in offdiagonals(column)
            l_val = le.ansatz(l, params)
            num += l_val * melem * k_val
        end

        (num, den)
    end

    numerator, denominator = result

    return numerator / denominator
end

function (le::LocalEnergyEvaluator{N,T})(F, G, params) where {N,T}
    if !isnothing(G)
        val, grad = val_and_grad(le, params)
        G .= grad
    else
        val = le(params)
    end
    if !isnothing(F)
        return val
    else
        return nothing
    end
end

function (le::LocalEnergyEvaluator{N,T})(params...) where {N,T}
    return le(params)
end
