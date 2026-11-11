"""
    val_err_grad_and_metric(f, params; kwargs...) -> (val, err, grad, S)

Return the energy `val` of `f` at `params`, its statistical error `err`, its gradient `grad`
with respect to the parameters, and the metric tensor (the S matrix)

```math
S_{kl} = ⟨O_k O_l⟩ - ⟨O_k⟩⟨O_l⟩, \\quad O_k(|f⟩) = \\frac{∂}{∂p_k} \\ln ψ(|f⟩),
```

where the expectation values are taken over ``|ψ|^2``.

Methods are implemented for [`LocalEnergyEvaluator`](@ref), which computes everything
exactly (and returns `err = 0`), and for [`KineticVQMC`](@ref), which estimates everything
from one sampling run. In the latter case, `kwargs` are passed to the `KineticVQMC` call.

Used by [`stochastic_reconfiguration`](@ref).
"""
function val_err_grad_and_metric(
    le::LocalEnergyEvaluator{N,T}, params; kwargs...
) where {N,T}
    p = SVector{N,T}(params)
    val, grad = val_and_grad(le, p)

    # Sums of ψ², ψ ∂ψ and ∂ψ ∂ψᵀ over the basis.
    init = (zero(T), zero(SVector{N,T}), zero(SMatrix{N,N,T}))
    norm_sq, ψ_dψ, dψ_dψ = Folds.mapreduce((a, b) -> map(+, a, b), le.basis; init) do addr
        ψ, dψ = val_and_grad(le.ansatz, addr, p)
        (abs2(ψ), ψ * dψ, dψ * dψ')
    end
    mean_O = ψ_dψ / norm_sq
    S = dψ_dψ / norm_sq - mean_O * mean_O'

    return val, zero(T), grad, S
end

function val_err_grad_and_metric(ke::KineticVQMC, params; kwargs...)
    ke(params; kwargs...)
    val, grad, S = _sampled_val_grad_and_metric(ke.result)
    err = local_energy_estimator(ke.result).err
    return val, err, grad, S
end

"""
    _sampled_val_grad_and_metric(res::KineticVQMCResult)

Estimate the energy, its gradient and the metric tensor from the samples of all walkers,
weighted by their residence times.
"""
function _sampled_val_grad_and_metric(res::KineticVQMCResult)
    add_tuples(a, b) = map(+, a, b)

    # First pass: means of the local energy and of the gradient ratios.
    first_pass = Folds.map(res.states) do st
        times = st.residence_times
        (
            sum(times),
            sum(i -> times[i] * st.local_energies[i], eachindex(times)),
            sum(i -> times[i] * st.grad_ratios[i], eachindex(times)),
        )
    end
    total_time, sum_e, sum_O = reduce(add_tuples, first_pass)
    val = sum_e / total_time
    mean_O = sum_O / total_time

    # Second pass: covariances, computed from deviations to avoid cancellation.
    second_pass = Folds.map(res.states) do st
        times = st.residence_times
        mapreduce(add_tuples, eachindex(times)) do i
            dO = st.grad_ratios[i] - mean_O
            (times[i] * dO * (st.local_energies[i] - val), times[i] * dO * dO')
        end
    end
    sum_force, sum_S = reduce(add_tuples, second_pass)
    grad = 2 * sum_force / total_time
    S = sum_S / total_time

    return val, grad, S
end

"""
    StochasticReconfigurationResult

Return type of [`stochastic_reconfiguration`](@ref). Holds the trace of the optimization.

Can be used as a `Table` from Tables.jl, e.g. converted to a `DataFrame`. Pass it in place
of the initial parameters to [`stochastic_reconfiguration`](@ref) to continue the
optimization.

# Fields

- `fun`: the optimized function.
- `hyperparameters::NamedTuple`: `τ` and `shift`.
- `initial_params`: the initial parameters.
- `iterations::Int`: number of iterations performed.
- `converged::Bool`, `reason::String`: whether and why the optimization stopped early.
- `param`, `value`, `error`, `gradient`, `param_delta`: vectors with the parameters, energy,
  its error, its gradient and the parameter update at each iteration.
"""
struct StochasticReconfigurationResult{T,P<:SVector{<:Any,T},F}
    fun::F
    hyperparameters::NamedTuple
    initial_params::P

    iterations::Int
    converged::Bool
    reason::String

    param::Vector{P}
    value::Vector{T}
    error::Vector{T}
    gradient::Vector{P}
    param_delta::Vector{P}
end

function Base.show(io::IO, r::StochasticReconfigurationResult)
    println(io, "StochasticReconfigurationResult")
    println(io, "  iterations: ", r.iterations)
    println(io, "  converged: ", r.converged, " (", r.reason, ")")
    println(io, "  last value: ", r.value[end], " ± ", r.error[end])
    print(io, "  last params: ", r.param[end])
end

Tables.istable(::Type{<:StochasticReconfigurationResult}) = true
Tables.rowaccess(::Type{<:StochasticReconfigurationResult}) = true
Tables.rows(r::StochasticReconfigurationResult) = StochasticReconfigurationResultRows(r)

"""
    StochasticReconfigurationResultRows

[Tables.jl](https://github.com/JuliaData/Tables.jl/)-compatible row iterator for
[`StochasticReconfigurationResult`](@ref).
"""
struct StochasticReconfigurationResultRows{T,P,F} <: AbstractVector{NamedTuple}
    result::StochasticReconfigurationResult{T,P,F}
end
function Base.getindex(rows::StochasticReconfigurationResultRows, i)
    r = rows.result
    return (;
        r.hyperparameters...,
        iter=i,
        param=r.param[i],
        value=r.value[i],
        error=r.error[i],
        gradient=r.gradient[i],
        param_delta=r.param_delta[i],
    )
end
Base.size(rows::StochasticReconfigurationResultRows) = (length(rows.result.param),)

"""
    stochastic_reconfiguration(
        f, p_init;
        maxiter=100, τ=0.05, shift=1e-3, fix_params=nothing, verbose=true,
        early_stop=true, param_tol=√eps(T), grad_tol=√eps(T), fkwargs=(;),
    )

Minimize the energy of an ansatz with stochastic reconfiguration [1]. In each iteration, the
parameters are updated by

```math
δ𝐩 = -τ (S + ε \\, \\mathrm{diag}(S) + ε_0)^{-1} 𝐟, \\quad 𝐟 = ∇E / 2,
```

where ``S`` is the metric tensor (see [`val_err_grad_and_metric`](@ref)) and ``ε`` is the
relative diagonal `shift`, which regularizes ``S`` when it is singular or noisy. A small
absolute shift ``ε_0`` is added to keep the update finite for parameters that do not affect
the ansatz. Unlike plain gradient descent, the update measures step sizes by how much the
wave function changes, which makes it robust to badly scaled or redundant parameters.

`f` can be a [`KineticVQMC`](@ref), which estimates all quantities by sampling, or a
[`LocalEnergyEvaluator`](@ref), which computes them exactly. Returns a
[`StochasticReconfigurationResult`](@ref).

# Arguments

- `f`: the function to optimize. Must support [`val_err_grad_and_metric`](@ref).
- `p_init`: initial parameters. Alternatively, a [`StochasticReconfigurationResult`](@ref)
  may be passed to continue a previous optimization from its [`final_params`](@ref).

# Keyword arguments

- `maxiter=100`: maximum number of iterations.
- `τ=0.05`: step size, similar to an imaginary time step. Larger values converge faster,
  but can become unstable, especially with several parameters.
- `shift=1e-3`: relative diagonal shift ``ε`` used to regularize ``S``.
- `fix_params`: a vector or tuple of booleans. If `fix_params[i]` is `true`, the `i`-th
  parameter is kept at its initial value.
- `verbose=true`: show a progress bar.
- `early_stop=true`: stop when the norm of the parameter update is below `param_tol`, or the
  norm of the gradient is below `grad_tol`. With sampling noise, these are rarely
  reached, so check convergence by plotting the trace.
- `fkwargs=(;)`: keyword arguments passed to `f`, e.g. `(; samples=1e5)` for a
  [`KineticVQMC`](@ref).

[1]: [S. Sorella, Phys. Rev. B **64**, 024512 (2001), doi:10.1103/PhysRevB.64.024512](https://doi.org/10.1103/PhysRevB.64.024512)
"""
function stochastic_reconfiguration(f, params; kwargs...)
    return stochastic_reconfiguration(f, SVector{length(params)}(float.(params)); kwargs...)
end
function stochastic_reconfiguration(f, prev::StochasticReconfigurationResult; kwargs...)
    return stochastic_reconfiguration(f, final_params(prev); kwargs...)
end

"""
    final_params(res::StochasticReconfigurationResult)

Return the parameters after the last update of the optimization, i.e.
`res.param[end] + res.param_delta[end]`. Note that `res.param[end]` holds the parameters at
which the last value and gradient were evaluated, before the last update. If the last update
was not finite, `res.param[end]` is returned.
"""
function final_params(res::StochasticReconfigurationResult)
    δp = res.param_delta[end]
    return all(isfinite, δp) ? res.param[end] + δp : res.param[end]
end
function stochastic_reconfiguration(
    f, p_init::P;
    maxiter=100, τ=0.05, shift=1e-3, fix_params=nothing, verbose=true,
    early_stop=true, param_tol=√eps(T), grad_tol=√eps(T), fkwargs=(;),
) where {N,T,P<:SVector{N,T}}
    if isnothing(fix_params)
        free = @SVector fill(true, N)
    else
        free = .!SVector{N,Bool}(fix_params)
    end
    # Restrict the linear system to the free parameters; fixed ones get a zero update.
    free_mask = SMatrix{N,N,T}(free * free')
    fixed_diag = Diagonal(SVector{N,T}(.!free))

    params = P[]
    values = T[]
    errors = T[]
    gradients = P[]
    param_deltas = P[]
    converged = false
    reason = "iterations"

    p = p_init
    verbose && (prog = Progress(maxiter))
    for iter in 1:maxiter
        val, err, grad, S = val_err_grad_and_metric(f, p; fkwargs...)
        grad = grad .* free

        S_reg = S + shift * Diagonal(diag(S)) + 1e-10 * I
        S_reg = S_reg .* free_mask + fixed_diag
        δp = -τ * (S_reg \ (grad / 2))

        push!(params, p)
        push!(values, val)
        push!(errors, err)
        push!(gradients, grad)
        push!(param_deltas, δp)

        if !all(isfinite, δp)
            reason = "non-finite parameter update"
            verbose && @warn "Stopped after $iter iterations: $reason"
            break
        elseif early_stop && norm(δp) < param_tol
            verbose && @info "Converged after $iter iterations (params)"
            reason = "params"
            converged = true
            break
        elseif early_stop && norm(grad) < grad_tol
            verbose && @info "Converged after $iter iterations (gradient)"
            reason = "gradient"
            converged = true
            break
        end

        p = p + δp
        verbose && next!(
            prog; showvalues=((:iter, iter), (:value, val ± err), (:param, p))
        )
    end
    verbose && finish!(prog)

    return StochasticReconfigurationResult(
        f, (; τ, shift), p_init, length(params), converged, reason,
        params, values, errors, gradients, param_deltas,
    )
end
