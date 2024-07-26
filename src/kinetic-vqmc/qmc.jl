"""
    kinetic_sample!(prob_buffer, hamiltonian, ansatz, curr_addr)

Use the `hamiltonian` and `curr_addr` to sample a new address from `ansatz`, while also
computing the local energy contribution of `curr_addr`. Return a tuple of the new address,
the residence time of the current address, and the local energy.

`prob_buffer` is a `Vector{Float64}` used as temporary storage.

The algorithm for sampling is taken from [1].

[1]: https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.8b00780
"""
function kinetic_sample!(prob_buffer, hamiltonian, ansatz, params, addr1)
    offdiags = offdiagonals(hamiltonian, addr1)

    resize!(prob_buffer, length(offdiags))

    val1, grad1 = val_and_grad(ansatz, addr1, params)
    diag = diagonal_element(hamiltonian, addr1)
    local_energy_num = diag * val1

    residence_time_denom = 0.0

    for (i, (addr2, melem)) in enumerate(offdiags)
        val2, grad2 = val_and_grad(ansatz, addr2, params)
        residence_time_denom += abs(val2)
        local_energy_num += melem * val2

        prob_buffer[i] = residence_time_denom
    end

    residence_time = abs(val1) / residence_time_denom
    if !isfinite(residence_time)
        error("Infinite residence time at $params")
    end
    local_energy = local_energy_num / val1
    gradient = grad1 / val1

    chosen = pick_random_from_cumsum(prob_buffer)
    new_addr, _ = offdiags[chosen]

    return new_addr, residence_time, local_energy, gradient
end

"""
    pick_random_from_cumsum(cumsum)

Pick a random index from `cumsum`, which should contain a cumulative sum (see `Base.cumsum`)
of values proportional to the probabilities with which the index should be picked.
"""
@inline function pick_random_from_cumsum(cumsum)
    chosen = rand() * last(cumsum)
    i = 1
    @inbounds while true
        if chosen < cumsum[i]
            return i
        end
        i += 1
    end
end

"""
    kinetic_vqmc!(st::KineticVQMCWalkerState; steps)
    kinetic_vqmc!(sts::Vector{KineticVQMCWalkerState}; steps)
    kinetic_vqmc!(res::KineticVQMCResult; steps)

Continue a [`kinetic_vqmc`](@ref) computation, peforming `steps` additional steps
(default is the current number of samples in the input).

Returns its input.
"""
function kinetic_vqmc!(st::KineticVQMCWalkerState; steps=length(st))
    curr_addr = st.curr_address
    @unpack (residence_times, local_energies, grad_ratios, addresses, prob_buffer) = st

    first_step = length(residence_times) + 1
    last_step = length(residence_times) + Int(steps)

    resize!(addresses, last_step)
    resize!(residence_times, last_step)
    resize!(local_energies, last_step)
    resize!(grad_ratios, last_step)

    for k in first_step:last_step
        next_addr, residence_time, local_energy, grad = kinetic_sample!(
            prob_buffer, st.hamiltonian, st.ansatz, st.params, curr_addr
        )
        addresses[k] = curr_addr
        residence_times[k] = residence_time
        local_energies[k] = local_energy
        grad_ratios[k] = grad
        curr_addr = next_addr # local energy was calculated for the previous address.
    end
    st.curr_address = curr_addr
    return st
end
# Parallel version
function kinetic_vqmc!(
    states::Vector{<:KineticVQMCWalkerState}; steps=length(states[1]),
)
    Threads.@threads for walker in eachindex(states)
        kinetic_vqmc!(states[walker]; steps)
    end
    return states
end
# Continuation version
function kinetic_vqmc!(res::KineticVQMCResult; steps=length(res.states[1]))
    kinetic_vqmc!(res.states; steps)
    return res
end

"""
    kinetic_vqmc(hamiltonian, ansatz, params; steps=1e6, walkers=Threads.nthreads() * 2)

Use continuous time variational quantum Monte Carlo to sample a addresses and local energies
from `hamiltonian` and `ansatz`. The `ansatz` must be indexable by addresses accepted by
`hamiltonian` and must support `Base.keytype` and `Base.valtype`.

The sampling samples `steps` steps per `walker`. The walkers are independent runs that run
on separate threads.

The function returns a [`KineticVQMCResult`](@ref)s, which contains the states of all
walkers. To continue a run, use [`continous_time_vqmc!`](@ref).

See [I. Sabzevari and S. Sharma](https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.8b00780) for
a detailed description of the algorithm.
"""
function kinetic_vqmc(
    hamiltonian, ansatz, params;
    samples=1e6,
    walkers=Threads.nthreads() > 1 ? Threads.nthreads() * 2 : 1,
    steps=round(Int, samples / walkers),
    )

    if walkers > 1
        state = [KineticVQMCWalkerState(hamiltonian, ansatz, params) for _ in 1:walkers]
        states = state
    else
        state = KineticVQMCWalkerState(hamiltonian, ansatz, params)
        states = [state]
    end
    kinetic_vqmc!(state; steps)

    return KineticVQMCResult(states)
end
