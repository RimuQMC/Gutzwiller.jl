"""
    GrossPitaevskiiAnsatz(h::AbstractHamiltonian)
    GrossPitaevskiiAnsatz(address::BoseFS; valtype::Type = Float64) <: AbstractAnsatz

Ansatz representing a Gross-Pitaevskii product state of ``N`` bosons in ``M`` modes,
with single-particle orbital parameter vector ``𝐜 = (c_1, …, c_M)``:

```math
|Ψ_{\\mathrm{GP}}⟩ = \\frac{1}{\\sqrt{N!}} \\left( ∑_{m=1}^M cₘ aₘ^† \\right)^N |0⟩
```

Projected onto a Fock basis state ``|n_1, …, n_M⟩``, the amplitude is:

```math
⟨n₁, …, n_M | Ψ_{\\mathrm{GP}}⟩ = \\sqrt{\\frac{N!}{\\prod_{m=1}^M nₘ!}} \\prod_{m=1}^M cₘ^{nₘ}
```

# Example
```jldoctest
julia> H = HubbardReal1D(BoseFS((2, 0)))
julia> gpe = GrossPitaevskiiAnsatz(H)
julia> gpe_from_address = GrossPitaevskiiAnsatz(starting_address(H))
julia> complex_gpe = GrossPitaevskiiAnsatz(starting_address(H); valtype=ComplexF64)

julia> params = SVector(inv(sqrt(2.0)), inv(sqrt(2.0)))
julia> addr = BoseFS((1, 1))
julia> amplitude = gpe(addr, params)
julia> amplitude, gradient = val_and_grad(gpe, addr, params)

julia> basis = build_basis(gpe)
julia> state = PDVec(gpe, params; basis=basis)

julia> evaluator = LocalEnergyEvaluator(H, gpe)
julia> energy = evaluator(params)
julia> energy, energy_gradient = val_and_grad(evaluator, params);
```
"""
struct GrossPitaevskiiAnsatz{A,T<:Number,M} <: AbstractAnsatz{A,T,M}
    address::A
end

function GrossPitaevskiiAnsatz(addr::BoseFS; valtype::Type=Float64)
    return GrossPitaevskiiAnsatz{typeof(addr),valtype,num_modes(addr)}(addr)
end

GrossPitaevskiiAnsatz(h::AbstractHamiltonian) =
    GrossPitaevskiiAnsatz(starting_address(h); valtype=eltype(h))

Rimu.starting_address(gpe::GrossPitaevskiiAnsatz) = gpe.address
Rimu.build_basis(gpe::GrossPitaevskiiAnsatz) = build_basis(gpe.address)

# evaluate GP ansatz amplitude
function (gpe::GrossPitaevskiiAnsatz{A,T,M})(addr::BoseFS{<:Any,M}, params) where {A,T,M}
    orb_prod = one(promote_type(T, eltype(params)))
    for (k, m, _) in occupied_modes(addr)
        c_m = params[m]
        iszero(c_m) && return zero(orb_prod)
        orb_prod *= c_m^k
    end
    # return coefficient sqrt(N! / prod(n_m!)) * prod(c_m^n_m)
    return sqrt(gamma(num_particles(addr) + 1) * multinomial_weight(addr)) * orb_prod
end

function val_and_grad(
    gpe::GrossPitaevskiiAnsatz{A,T,M}, addr::BoseFS{<:Any,M}, params
) where {A,T,M}
    # <addr|Ψ_GP>
    val = gpe(addr, params)
    # prefactor
    coefficient = sqrt(gamma(num_particles(addr) + 1) * multinomial_weight(addr))
    grad = zeros(SVector{M,typeof(val)})
    # derivative
    for (k, m, _) in occupied_modes(addr)
        # product over remaining occupied modes
        other_prod = prod(
            (params[j]^l for (l, j, _) in occupied_modes(addr) if j != m);
            init=one(val),
        )
        # partial derivative ∂<addr|Ψ_GP>/∂c_m
        grad = setindex(grad, coefficient * k * params[m]^(k - 1) * other_prod, m)
    end

    return val, grad
end
