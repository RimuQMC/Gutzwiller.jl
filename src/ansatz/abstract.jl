"""
    AbstractAnsatz{K,V,N}

Abstract type for ansatzes. An ansatz is a function that maps a Fock basis state to a value,
with a set of parameters, and thus represents a quantum state. Variational Monte Carlo (VMC)
methods can be used to optimize the parameters of an ansatz to minimize the energy or the
variance of the energy of a given Hamiltonian. The ansatz can be also be used as a guiding
function for importance sampling using [`AnsatzSampling`](@ref) in a projector Monte Carlo
simulation with [`ProjectorMonteCarloProblem`](@extref Rimu.ProjectorMonteCarloProblem), or
to compute expectation values of observables.

An ansatz has a `keytype`
[`K <: AbstractFockAddress`](@extref Rimu Rimu.Interfaces.AbstractFockAddress) representing
the type of the Fock basis states, and `valtype` `V <: Number` representing the type of the
values it produces. It has `N` parameters.

It behaves similar to an [`AbstractDVec`](@extref Rimu Rimu.Interfaces.AbstractDVec) with
`keytype` `K` and `valtype` `V`.

## Implemented subtypes
* [`GutzwillerAnsatz`](@ref)
* [`ExtendedGutzwillerAnsatz`](@ref)
* [`VectorAnsatz`](@ref)
* [`MultinomialAnsatz`](@ref)
* [`JastrowAnsatz`](@ref)
* [`RelativeJastrowAnsatz`](@ref)
* [`DensityProfileAnsatz`](@ref)
* [`CombinationAnsatz`](@ref)
* [`GrossPitaevskiiAnsatz`](@ref)

# Extended Help
Define your own ansatz by subtyping `MyAnsatz <: AbstractAnsatz` and implementing the
following methods for an instance `ansatz::MyAnsatz` of your type:

* `ansatz(key::K, params)::V`: Get the value of the ansatz for a given address `key` with
  specified parameters.
* [`val_and_grad(ansatz, key, params)`](@ref): Get the value and gradient (w.r.t. the
  parameters) of the ansatz.
* [`build_basis(ansatz)`](@extref Rimu Rimu.ExactDiagonalization.build_basis): for
  collecting the vector to [`DVec`](@extref Rimu Rimu.DictVectors.DVec) or
  [`PDVec`](@extref Rimu Rimu.DictVectors.PDVec) (optional).
"""
abstract type AbstractAnsatz{K,V,N} end

Base.keytype(::AbstractAnsatz{K}) where {K} = K
Base.keytype(::Type{<:AbstractAnsatz{K}}) where {K} = K
Base.valtype(::AbstractAnsatz{<:Any,V}) where {V} = V
Base.valtype(::Type{<:AbstractAnsatz{<:Any,V}}) where {V} = V
num_parameters(::Type{<:AbstractAnsatz{<:Any,<:Any,N}}) where {N} = N
num_parameters(::AbstractAnsatz{<:Any,<:Any,N}) where {N} = N

function collect_to_vec!(dst, ans::AbstractAnsatz, params, basis)
    for k in basis
        dst[k] = ans(k, params)
    end
    return dst
end
function Rimu.DVec(
    ans::AbstractAnsatz{K,V,N}, params; basis=build_basis(ans), kwargs...
) where {K,V,N}
    result = DVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans, SVector{N,V}(params), basis)
end
function Rimu.PDVec(
    ans::AbstractAnsatz{K,V,N}, params; basis=build_basis(ans), kwargs...
) where {K,V,N}
    result = PDVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans, SVector{N,V}(params), basis)
end

"""
    val_and_grad(::AbstractAnsatz, addr, params)

Return ansatz value at `addr` and its gradient w.r.t. `params`.

See also [`AbstractAnsatz`](@ref).
"""
val_and_grad

function val_and_grad(a::AbstractAnsatz{K,<:Any,0}, addr::K, _) where {K}
    return a(addr, SVector{0,valtype(a)}()), SVector{0,valtype(a)}()
end

"""
    val_err_and_grad(args...)

Return the value, its error and gradient. See [`val_and_grad`](@ref).
"""
function val_err_and_grad(args...)
    val, grad = val_and_grad(args...)
    return val, zero(typeof(val)), grad
end
