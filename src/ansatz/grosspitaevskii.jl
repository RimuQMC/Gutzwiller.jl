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
julia> gpa = GrossPitaevskiiAnsatz(BoseFS(2, 0))
GrossPitaevskiiAnsatz(BoseFS(2, 0))

julia> amplitude = gpa(BoseFS(1,1), [1.0, 2.0])
2.8284271247461903
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

Rimu.starting_address(gpa::GrossPitaevskiiAnsatz) = gpa.address
Rimu.build_basis(gpa::GrossPitaevskiiAnsatz) = build_basis(gpa.address)

function Base.show(io::IO, gpa::GrossPitaevskiiAnsatz{A,V,N}) where {A,V,N}
    print(io, "GrossPitaevskiiAnsatz($(gpa.address))")
end

# evaluate GP ansatz amplitude
function (gpa::GrossPitaevskiiAnsatz{A,T,M})(addr::BoseFS{<:Any,M}, params) where {A,T,M}
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
    gpa::GrossPitaevskiiAnsatz{A,T,M}, addr::BoseFS{<:Any,M}, params
) where {A,T,M}
    # <addr|Ψ_GP>
    val = gpa(addr, params)
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
