"""
    kinetic_sample!(
        offdiag_buffer, prob_buffer, hamiltonian, adjoint, ansatz, params, curr_addr
    )

Use the `hamiltonian` and `curr_addr` to sample a new address from `ansatz` with parameters
`params`, while also computing the local energy and the gradient ratio of `curr_addr`.

`adjoint` is `nothing` for a Hermitian `hamiltonian`, and its adjoint otherwise. In the
latter case, the local energy is computed for the Hermitian part of the `hamiltonian`
(see [`_symmetrize_offdiagonal`](@ref)).

Return a tuple of

* the new address,
* the residence time of `curr_addr`,
* the local energy of `curr_addr`,
* the gradient of the ansatz w.r.t. `params` at `curr_addr`, divided by its value.

`offdiag_buffer` is a `Vector{Pair{A,T}}` and `prob_buffer` is a `Vector{T}`, both used as
temporary storage.

The algorithm for sampling is taken from [1].
This function is used internally by [`kinetic_vqmc!`](@ref) and [`kinetic_vqmc`](@ref).

[1]: [I. Sabzevari and S. Sharma, JCTC (2018), doi:10.1021/acs.jctc.8b00780](https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.8b00780)
"""
function kinetic_sample!(
    offdiags, prob_buffer, hamiltonian, adjoint, ansatz, params, addr_n
)
    column = hamiltonian * addr_n
    resize!(offdiags, num_offdiagonals(column))
    resize!(prob_buffer, num_offdiagonals(column))

    for (i, (k, v)) in enumerate(offdiagonals(column))
        offdiags[i] = k => v
    end

    val_n, grad_n = val_and_grad(ansatz, addr_n, params)
    if iszero(val_n) || !isfinite(val_n)
        # The walker never moves to an address with zero or non-finite amplitude, so this
        # can only happen at the starting address.
        error(
            "Ansatz value at $addr_n is $val_n with params=$params. The starting address " *
            "must have a finite, non-zero amplitude."
        )
    end
    diag = diagonal_element(column)
    offdiag_num = zero(val_n)

    residence_time_denom = 0.0

    for (i, (addr_m, offdiag)) in enumerate(offdiags)
        val_m = ansatz(addr_m, params)
        residence_time_denom += abs(val_m)
        offdiag_num += offdiag * val_m

        prob_buffer[i] = residence_time_denom
    end

    if !isfinite(residence_time_denom)
        error("Non-finite ansatz values at neighbours of $addr_n with params=$params.")
    elseif iszero(residence_time_denom)
        error("All neighbours of $addr_n have zero amplitude with params=$params.")
    end

    residence_time = abs(val_n) / residence_time_denom
    offdiag_num = _symmetrize_offdiagonal(adjoint, ansatz, params, addr_n, offdiag_num)
    local_energy = diag + offdiag_num / val_n
    gradient = grad_n / val_n

    chosen = pick_random_from_cumsum(prob_buffer)
    addr_m, _ = offdiags[chosen]

    return addr_m, residence_time, local_energy, gradient
end

"""
    _symmetrize_offdiagonal(adjoint, ansatz, params, addr_n, offdiag_num)

Return the off-diagonal part of the local energy numerator for the Hermitian part of the
Hamiltonian, ``(H + H^†)/2``.

`offdiag_num` is ``∑_m ⟨m|H|n⟩ ψ(m)``, computed from the column of the Hamiltonian at
`addr_n`. For a Hermitian Hamiltonian (`adjoint === nothing`), this is returned unchanged.
Otherwise, it is averaged with ``∑_m ⟨n|H|m⟩ ψ(m)``, computed from the column of `adjoint`.
Using the Hermitian part keeps the expected local energy equal to the Rayleigh quotient,
and makes the gradient estimator in [`val_and_grad`](@ref) valid.
"""
_symmetrize_offdiagonal(::Nothing, _, _, _, offdiag_num) = offdiag_num
function _symmetrize_offdiagonal(adjoint, ansatz, params, addr_n, offdiag_num)
    adjoint_num = zero(offdiag_num)
    for (addr_m, offdiag) in offdiagonals(adjoint * addr_n)
        adjoint_num += conj(offdiag) * ansatz(addr_m, params)
    end
    return (offdiag_num + adjoint_num) / 2
end

"""
    pick_random_from_cumsum(cumsum)

Pick a random index from `cumsum`, which should contain a cumulative sum (see `Base.cumsum`)
of values proportional to the probabilities with which the index should be picked.

`last(cumsum)` must be finite and positive. Indices with zero probability are never picked.
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

Continue a [`kinetic_vqmc`](@ref) computation, performing `steps` additional steps
(default is the current number of samples in the input).

Returns its input.
"""
function kinetic_vqmc!(st::KineticVQMCWalkerState; steps=length(st))
    curr_addr = st.curr_address
    @unpack (
        residence_times, local_energies, grad_ratios, addresses, prob_buffer, offdiag_buffer
    ) = st

    first_step = length(residence_times) + 1
    last_step = length(residence_times) + Int(steps)

    resize!(addresses, last_step)
    resize!(residence_times, last_step)
    resize!(local_energies, last_step)
    resize!(grad_ratios, last_step)

    for k in first_step:last_step
        next_addr, residence_time, local_energy, grad = kinetic_sample!(
            offdiag_buffer, prob_buffer, st.hamiltonian, st.adjoint, st.ansatz, st.params,
            curr_addr,
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
walkers. To continue a run, use [`kinetic_vqmc!`](@ref).

See [I. Sabzevari and S. Sharma, JCTC (2018), doi:10.1021/acs.jctc.8b00780](https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.8b00780)
for a detailed description of the algorithm.

See also [`AbstractAnsatz`](@ref).
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
