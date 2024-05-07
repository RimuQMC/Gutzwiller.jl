"""


```math
b
```
"""
struct MultinomialAnsatz{K} <: Gutzwiller.AbstractAnsatz{K,Float64,1}
    normalization::Float64
end
function MultinomialAnsatz(H::AbstractHamiltonian; normalize=false)
    addr = starting_address(H)
    K = typeof(addr)
    N = float(num_particles(K))
    if normalize
        normalization = multinomial_normalization(addr)
    else
        normalization = 1.0
    end
    return MultinomialAnsatz{K}(normalization)
end

function multinomial_normalization(addr::BoseFS)
    return gamma(num_particles(addr) + 1) / float(num_modes(addr))^num_particles(addr)
end
function multinomial_normalization(addr::FermiFS)
    return 1.0
end
function multinomial_normalization(addr::CompositeFS)
    return prod(multinomial_normalization, addr.components)
end

function multinomial_weight(addr::BoseFS, occ=OccupiedModeMap(addr))
    return prod(idx -> 1/gamma(idx.occnum + 1), occ)
end
multinomial_weight(addr::FermiFS) = 1.0
multinomial_weight(addr::CompositeFS) = multinomial_weight(addr.components)
multinomial_weight(addrs::Tuple{}) = 1.0
multinomial_weight((a, as...)::Tuple) = multinomial_weight(a) * multinomial_weight(as)

function (b::MultinomialAnsatz)(addr, p)
    return (multinomial_weight(addr) * b.normalization)^p[1]
end
function val_and_grad(b::MultinomialAnsatz, addr, param)
    p = only(param)
    y = multinomial_weight(addr) * b.normalization
    val = y^p
    val, SVector{1,Float64}(val * log(y))
end
