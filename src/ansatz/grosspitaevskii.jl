"""
    GrossPitaevskiiAnsatz(address::BoseFS, valtype::Type = Float64) <: AbstractAnsatz

Ansatz representing a Gross-Pitaevskii product state of ``N`` bosons in ``M`` modes,
with single-particle orbital parameter vector ``𝐜 = (c_1, …, c_M)``:

```math
|Ψ_{\\mathrm{GP}}⟩ = \\frac{1}{\\sqrt{N!}} \\left( ∑_{m=1}^M cₘ aₘ^† \\right)^N |0⟩
```

Projected onto a Fock basis state ``|n_1, …, n_M⟩``, the amplitude is:

```math
⟨n₁, …, n_M | Ψ_{\\mathrm{GP}}⟩ = \\sqrt{\\frac{N!}{\\prod_{m=1}^M nₘ!}} \\prod_{m=1}^M cₘ^{nₘ}
```
"""
struct GrossPitaevskiiAnsatz{A,V<:Number,N} <: AbstractAnsatz{A,V,N}
    address::A
end

function GrossPitaevskiiAnsatz(addr::BoseFS, valtype::Type)
    return GrossPitaevskiiAnsatz{typeof(addr),valtype,num_modes(addr)}(addr)
end

GrossPitaevskiiAnsatz(h::AbstractHamiltonian) = GrossPitaevskiiAnsatz(starting_address(h), eltype(h))
GrossPitaevskiiAnsatz(addr::BoseFS) = GrossPitaevskiiAnsatz(addr, Float64)

Rimu.starting_address(gpe::GrossPitaevskiiAnsatz) = gpe.address
Rimu.build_basis(gpe::GrossPitaevskiiAnsatz) = build_basis(gpe.address)

Base.show(io::IO, gpe::GrossPitaevskiiAnsatz{A,V,N}) where {A,V,N} =
    print(io, "GrossPitaevskiiAnsatz{$V, modes=$N}($(gpe.address))")

# evaluate GP ansatz amplitude
function (gpe::GrossPitaevskiiAnsatz{A,V,N})(addr::BoseFS{<:Any,N}, params::SVector{N}) where {A,V,N}
    orb_prod = one(promote_type(V, eltype(params)))
    for (k, m, _) in occupied_modes(addr)
        c_m = params[m]
        iszero(c_m) && return zero(orb_prod)
        orb_prod *= c_m^k
    end
    # return coefficient sqrt(N! / prod(n_m!)) * prod(c_m^n_m)
    return sqrt(gamma(num_particles(addr) + 1) * multinomial_weight(addr)) * orb_prod
end