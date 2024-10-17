"""
    KineticVQMC(
        hamiltonian, ansatz;
        samples=1e6,
        walkers=Threads.nthreads() > 1 ? Threads.nthreads() * 2 : 1,
        steps=round(Int, samples / walkers),
    )

Usage

```julia
julia> ham = HubbardReal1D(BoseFS((1,1,1,1)));

julia> qmc = KineticVQMC(ham, GutzwillerAnsatz(ham); samples=1e5);

julia> qmc(0.5)
-6.650547790454165

julia> val_and_grad(qmc, 0.5)
(-6.6525129909164304, [-0.04590881240854946])
```

The result of a run is stored in the struct and can be accessed and manipulated after a
computation. If `qmc` or `val_and_grad` is called again, the result is overwritten.

```julia
julia> qmc.result
KineticVQMCResult
  walkers:      24
  samples:      100008
  local energy: -6.6525 ± 0.001509

julia> local_energy_estimator(qmc.result)
CombinedBlockingResult{Float64}
  mean = -6.6525 ± 0.0015
  with uncertainty of ± 4.082709649722105e-5
  Combined from 24 blocking results. (k ∈ 2 … 6)

julia> val_and_grad(qmc.result)
(-6.6525129909164304, [-0.04590881240854946])

```

"""
struct KineticVQMC{N,T,H,A<:AbstractAnsatz{<:Any,T,N},R}
    hamiltonian::H
    ansatz::A
    steps::Int
    walkers::Int
    result::R
end
function KineticVQMC(
    hamiltonian::H, ansatz::A;
    samples=1e6,
    walkers=Threads.nthreads() > 1 ? Threads.nthreads() * 2 : 1,
    steps=round(Int, samples / walkers),
) where {H,T,N,A<:AbstractAnsatz{<:Any,T,N}}
    params = zeros(SVector{N,T})
    result = KineticVQMCResult(
        [KineticVQMCWalkerState(hamiltonian, ansatz, params) for _ in 1:walkers]
    )
    return KineticVQMC(hamiltonian, ansatz, steps, walkers, result)
end

function Base.show(io::IO, ke::KineticVQMC)
    println(io, "KineticVQMC(")
    println(io, "  ", ke.hamiltonian, ",")
    println(io, "  ", ke.ansatz, ";")
    println(io, "  steps=", ke.steps, ",")
    println(io, "  walkers=", ke.walkers, ",")
    print(io, ")")
end

function _reset!(ke::KineticVQMC{N,T}, params) where {N,T}
    foreach(ke.result.states) do st
        st.params = SVector{N,T}(params)
        empty!(st)
    end
end

function (ke::KineticVQMC)(params)
    _reset!(ke, params)
    res = kinetic_vqmc!(ke.result; steps=ke.steps)
    return mean(res)
end

function val_and_grad(ke::KineticVQMC, params)
    _reset!(ke, params)
    res = kinetic_vqmc!(ke.result; steps=ke.steps)
    return val_and_grad(res)
end
function val_err_and_grad(ke::KineticVQMC, params)
    _reset!(ke, params)
    res = kinetic_vqmc!(ke.result; steps=ke.steps)
    return val_err_and_grad(res)
end

function (ke::KineticVQMC)(F, G, params)
    _reset!(ke, params)
    res = kinetic_vqmc!(ke.result; steps=ke.steps)
    if !isnothing(G)
        val, grad = val_and_grad(res)
        G .= grad
    else
        val = mean(res)
    end
    if !isnothing(F)
        return val
    else
        return nothing
    end
end
