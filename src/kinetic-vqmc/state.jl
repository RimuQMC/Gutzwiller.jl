using Rimu.StatsTools: BlockingResult

"""
    mutable struct KineticVQMCWalkerState(hamiltonian, ansatz)

Contains all information needed to perform continuous time variational quantum Monte Carlo.

See also: [`kinetic_vqmc!`](@ref), [`kinetic_vqmc`](@ref).
"""
mutable struct KineticVQMCWalkerState{A,T<:Real,H,N,V<:AbstractAnsatz{A,T,N}}
    hamiltonian::H
    ansatz::V
    params::SVector{N,T}
    curr_address::A
    addresses::Vector{A}
    residence_times::Vector{T}
    local_energies::Vector{T}
    grad_ratios::Vector{SVector{N,T}}
    prob_buffer::Vector{T}
    offdiag_buffer::Vector{Pair{A,T}}
end
function KineticVQMCWalkerState(
    hamiltonian::H, ansatz::V, params
) where {H,A,T,N,V<:AbstractAnsatz{A,T,N}}
    return KineticVQMCWalkerState{A,T,H,N,V}(
        hamiltonian, ansatz, SVector{N,T}(params),
        starting_address(hamiltonian), A[], T[], T[], T[], T[], Pair{A,T}[],
    )
end

Base.keytype(::KineticVQMCWalkerState{A}) where {A} = A
Base.valtype(::KineticVQMCWalkerState{<:Any,T}) where {T} = T

function Base.empty!(st::KineticVQMCWalkerState)
    empty!(st.residence_times)
    empty!(st.local_energies)
    empty!(st.addresses)
    return st
end
function Base.length(st::KineticVQMCWalkerState)
    return length(st.addresses)
end
function val_and_grad(res::KineticVQMCWalkerState)
    weights = FrequencyWeights(res.residence_times)
    val = mean(res.local_energies, weights)
    grads = res.grad_ratios .* (res.local_energies .- val)
    grad = 2 * mean(grads, weights)

    return val, grad
end
function val_err_and_grad(res::KineticVQMCWalkerState)
    weights = FrequencyWeights(res.residence_times)

    b_res = blocking_analysis(resample(res.residence_times, res.local_energies))
    val = b_res.mean
    err = b_res.err

    grads = res.grad_ratios .* (res.local_energies .- val)
    grad = 2 * mean(grads, weights)
    return val, err, grad
end


"""
    KineticVQMCResult

The result of [`kinetic_vqmc`](@ref). Holds the state of each walker. Use
[`local_energy_estimator`](@ref)`(result)` to get an estimate of the local energy, or
[`PDVec`](@ref)`(result)` to materialize the sampled vector.

Supports the [Tables.jl](https://github.com/JuliaData/Tables.jl/) interface and a such can
be converted to a `DataFrame` or saved to file with
[Arrow.jl](https://github.com/apache/arrow-julia) or
[CSV.jl](https://github.com/JuliaData/CSV.jl).
"""
struct KineticVQMCResult{A,T,C<:KineticVQMCWalkerState{A,T}}
    states::Vector{C}
end
function Base.show(io::IO, res::KineticVQMCResult)
    est = local_energy_estimator(res)
    μ = round(est.mean; sigdigits=5)
    σ = round(est.err; sigdigits=5)
    println(io, "KineticVQMCResult")
    println(io, "  walkers:      ", length(res.states))
    println(io, "  samples:      ", sum(length, res.states))
    print(io, "  local energy: ", μ, " ± ", σ)
end

Base.keytype(::KineticVQMCResult{A}) where {A} = A
Base.valtype(::KineticVQMCResult{<:Any,T}) where {T} = T

Tables.istable(::Type{<:KineticVQMCResult}) = true
Tables.rowaccess(::Type{<:KineticVQMCResult}) = true
Tables.rows(r::KineticVQMCResult) = KineticVQMCResultRows(r)
function Tables.schema(r::KineticVQMCResult{A,T}) where {A,T}
    return Tables.Schema(
        (:walker, :step, :address, :residence_time, :local_energy),
        Tuple{Int,Int,A,T,T},
    )
end

const ROWTYPE{A,T} = @NamedTuple{
    walker::Int, step::Int, address::A, residence_time::T, local_energy::T
}

"""
    KineticVQMCResultRows

[Tables.jl](https://github.com/JuliaData/Tables.jl/)-compatible row iterator for
[`KineticVQMCResult`](@ref).
"""
struct KineticVQMCResultRows{
    A,T,C<:KineticVQMCResult{A,T}
} <: AbstractVector{ROWTYPE{A,T}}
    result::C
end
function Base.getindex(rows::KineticVQMCResultRows, i)
    walker = 1
    states = rows.result.states
    while i > length(states[walker])
        i -= length(states[walker])
        walker += 1
    end
    st = states[walker]
    address = st.addresses[i]
    residence_time = st.residence_times[i]
    local_energy = st.local_energies[i]

    return (; walker, step=i, address, residence_time, local_energy)
end
function Base.size(rows::KineticVQMCResultRows)
    return (sum(length, rows.result.states), 1)
end

###
### Blocking utils
###
function val_and_grad(res::KineticVQMCResult{<:Any,T}) where {T}
    vals = zero(T)
    grads = zero(eltype(res.states[1].grad_ratios))
    walkers = length(res.states)

    for w in 1:walkers
        v, g = val_and_grad(res.states[w])
        vals += v
        grads += g
    end

    return vals / walkers, grads ./ walkers
end
function val_err_and_grad(res::KineticVQMCResult{<:Any,T}) where {T}
    vals = zero(T)
    errs = zero(T)
    grads = zero(eltype(res.states[1].grad_ratios))
    walkers = length(res.states)

    for w in 1:walkers
        v, e, g = val_err_and_grad(res.states[w])
        errs += e^2
        vals += v
        grads += g
    end

    return vals / walkers, √mean(errs) / walkers, grads ./ walkers
end

"""
    resample(residence_times, values; len=length(values))

Resample unevenly spaced timeseries data into a vector of `len` evenly-spaced values.  Here,
the time `values[i]` occupies is equal to `residence_times[i]`. The resampling is done in
such a way that

```julia
    mean(resample(xs, ys; len)) == mean(xs, FrequencyWeights(ys))
```

independent of `len`.

Using this function allows us to use [`blocking_analysis`](@ref) on the resampled data.

See also: [`resample!`](@ref).
"""
function resample(residence_times, values; len=length(residence_times))
    result = zeros(len)
    return resample!(result, residence_times, values)
end

"""
    resample!(result, residence_times, values)

In-place version of [`resample`](@ref). The length of the output is equal to
`length(result)`.
"""
function resample!(result, residence_times, values)
    if length(residence_times) ≠ length(values)
        throw(ArgumentError("Lengths of `residence_times` and `values` do not match!"))
    end

    len = length(result)

    total_time = sum(residence_times)
    t = 0.0
    k = 1
    μ = 0.0
    val = 0.0

    i = 1
    while true
        # Skip over all next slots covered by leftovers from previous step
        while t ≥ i && i ≤ len
            result[i] = val
            i += 1
            μ -= val
        end
        # Go through enough intervals to fill in a slot
        while t < i && k ≤ length(values)
            val = values[k]
            step = len * residence_times[k] / total_time # amount of time occupied by y
            μ += val * min(step, i - t)
            t += step
            k += 1
        end

        i > len && break
        result[i] = μ
        μ = (t - i) * val # Leftover to be taken to the next step
        i += 1
    end

    return result
end

"""
    CombinedBlockingResult{T}

Combination of several independent blocking results.
"""
struct CombinedBlockingResult{T}
    mean::T
    err::T
    err_err::T
    results::Vector{BlockingResult{T}}
end
function Base.show(io::IO, res::CombinedBlockingResult{T}) where {T}
    println(io, "CombinedBlockingResult{$T}")
    println(io, "  mean = ", Measurements.measurement(res.mean, res.err))
    println(io, "  with uncertainty of ± ", res.err_err)

    n_res = length(res.results)
    min_k, max_k = extrema(r.k for r in res.results)
    if min_k < 0
        println(io, "  Blocking unsuccessful.")
    else
        println(io, "  Combined from $n_res blocking results. (k ∈ $min_k … $max_k)")
    end
end

"""
    local_energy_estimator(res::KineticVQMCResultRows; kwargs...)

Return the local energy estimator from the result of a [`kinetic_vqmc`](@ref) run.
Keyword arguments are passed to [`blocking_analysis`](@ref). Returns a
[`CombinedBlockingResult`](@ref).
"""
function local_energy_estimator(
    res::KineticVQMCResult;
    resample_length=length(first(res.states)),
    kwargs...
)
    μ = 0.0
    v = 0.0
    results = Folds.map(res.states) do st
        blocking_analysis(
            resample(st.residence_times, st.local_energies; len=resample_length);
            kwargs...,
        )
    end
    n = length(results)
    return CombinedBlockingResult(
        mean(r.mean for r in results),
        √(mean(r.err^2 for r in results) / n),
        √(mean(r.err_err^2 for r in results) / n),
        results,
    )
end

function Statistics.mean(res::KineticVQMCResult)
    s = Folds.sum(res.states) do st
        mean(st.local_energies, FrequencyWeights(st.residence_times))
    end
    return s / length(res.states)
end
function Statistics.var(res::KineticVQMCResult)
    s = Folds.sum(res.states) do st
        var(st.local_energies, FrequencyWeights(st.residence_times))
    end
    return s / length(res.states)
end

###
### Vectors
###
function _to_vec!(dv, res::KineticVQMCResult)
    for row in Tables.rows(res)
        k = row.address
        v = row.residence_time
        deposit!(dv, k, v, nothing)
    end
    return normalize!(dv)
end

function Rimu.DVec(res::KineticVQMCResult{A,T}; kwargs...) where {A,T}
    dv = DVec{A,T}(; kwargs...)
    return _to_vec!(dv, res)
end
function Rimu.PDVec(res::KineticVQMCResult{A,T}; kwargs...) where {A,T}
    dv = PDVec{A,T}(; kwargs...)
    return _to_vec!(dv, res)
end
