# # Gutzwiller
#
# [![Coverage Status](https://coveralls.io/repos/github/mtsch/Gutzwiller.jl/badge.svg?branch=master)](https://coveralls.io/github/mtsch/Gutzwiller.jl?branch=master)
# [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://rimuqmc.github.io/Gutzwiller.jl/)
#
# _importance sampling and variational Monte Carlo for
# [Rimu.jl](https://github.com/joachimbrand/Rimu.jl)_
#
# ## Installation
#
# Gutzwiller.jl is not yet registered. To install it, run
#
# ```julia
# import Pkg; Pkg.add("https://github.com/mtsch/Gutzwiller.jl")
# ```
#

# ## Usage guide
#
#
using Rimu
using Gutzwiller
using CairoMakie
using LaTeXStrings

# First, we set up a starting address and a Hamiltonian

addr = near_uniform(BoseFS{10,10})
H = HubbardReal1D(addr; u=2.0)

# In this example, we'll set up a Gutzwiller ansatz to importance-sample the Hamiltonian.

ansatz = GutzwillerAnsatz(H)

# An ansatz is a struct that given a set of parameters and an address, produces the value
# it would have if it was a vector.

ansatz(addr, [1.0])

# In addition, the function `val_and_grad` can be used to compute both the value and its
# gradient with respect to the parameters.

val_and_grad(ansatz, addr, [1.0])

# ### Deterministic optimization

# For effective importance sampling, we want the ansatz to be as good of an approximation to
# the ground state of the Hamiltonian as possible. As the value of the Rayleigh quotient of
# a given ansatz is always larger than the Hamiltonian's ground state energy, we can use
# an optimization algorithm to find the paramters that minimize its energy.

# When the basis of the Hamiltonian is small enough to fit into memory, it's best to use the
# `LocalEnergyEvaluator`

le = LocalEnergyEvaluator(H, ansatz)

# which can be used to evaulate the value of the Rayleigh quotient (or its gradient) for
# given parameters. In the case of the Gutzwiller ansatz, there is only one parameter.

le([1.0])

# Like before, we can use `val_and_grad` to also evaluate its gradient.

val_and_grad(le, [1.0])

# Now, let's plot the energy landscape for this particular case

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"p", ylabel=L"E")
    ps = range(0, 2; length=100)
    Es = [le([p]) for p in ps]
    lines!(ax, ps, Es)
    fig
end

# To find the minimum, pass `le` to `optimize` from Optim.jl

using Optim, NLSolversBase

opt_nelder = optimize(le, [1.0])

# To take advantage of the gradients, wrap the evaluator in `only_fg!`. This will
# usually reduce the number of steps needed to reach the minimum.

opt_lbgfs = optimize(only_fg!(le), [1.0])

# We can inspect the parameters and the value at the minimum as

opt_lbgfs.minimizer, opt_lbgfs.minimum

# ### Variational quantum Monte Carlo

# When the Hamiltonian is too large to store its full basis in memory, we can use
# variational QMC to sample addresses from the Hilbert space and evaluate their energy
# at the same time. An important paramter we have tune is the number `steps`. More steps
# will give us a better approximation of the energy, but take longer to evaluate.
# Not taking enough samples can also result in producing a biased result.
# Consider the following.

p0 = [1.0]
kinetic_vqmc(H, ansatz, p0; steps=1e2) #hide
@time kinetic_vqmc(H, ansatz, p0; steps=1e2)

#

@time kinetic_vqmc(H, ansatz, p0; steps=1e5)

#

@time kinetic_vqmc(H, ansatz, p0; steps=1e7)

# For this simple example, `1e2` steps gives an energy that is significantly higher, while
# `1e7` takes too long. `1e5` seems to work well enough. For more convenient evaluation, we
# wrap VQMC into a struct that behaves much like the `LocalEnergyEvaluator`.

qmc = KineticVQMC(H, ansatz; samples=1e4)

#

qmc([1.0]), le([1.0])

# Because the output of this procedure is noisy, optimizing it with Optim.jl will not work.
# However, we can use a stochastic gradient descent (in this case
# [AMSGrad](https://paperswithcode.com/method/amsgrad)).

grad_result = amsgrad(qmc, [1.0])

# While `amsgrad` attempts to determine if the optimization converged, it will generally not
# detect convergence due to the noise in the QMC evaluation. The best way to determine
# convergence is to plot the results. `grad_result` can be converted to a `DataFrame`.

grad_df = DataFrame(grad_result)

# To plot it, we can either work with the `DataFrame` or access the fields directly

begin
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xlabel=L"i", ylabel=L"E_v")
    ax2 = Axis(fig[2, 1]; xlabel=L"i", ylabel=L"p")

    lines!(ax1, grad_df.iter, grad_df.value)
    lines!(ax2, first.(grad_result.param))
    fig
end

# We see from the plot that the value of the energy is fluctiating around what appears to be
# the minimum. While the parameter estimate here is probably good enough for importance
# sampling, we can refine the result by creating a new `KineticVQMC` structure with
# increased samples and use it to refine the result. Here, we can pass the previous result
# `grad_result` in place of the initial parameters, which will continue the computation
# where the previous one left off. Alternatively, this can be achieved by passing the `first_moment_init` and `second_moment_init` arguments to `amsgrad`.

qmc2 = KineticVQMC(H, ansatz; samples=1e6)
grad_result2 = amsgrad(qmc2, grad_result)

# Now, let's plot the refined result next to the minimum found by Optim.jl

begin
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xlabel=L"i", ylabel=L"E_v")
    ax2 = Axis(fig[2, 1]; xlabel=L"i", ylabel=L"p")

    lines!(ax1, grad_result2.value)
    hlines!(ax1, [opt_lbgfs.minimum]; linestyle=:dot)
    lines!(ax2, first.(grad_result2.param))
    hlines!(ax2, opt_lbgfs.minimizer; linestyle=:dot)
    fig
end

# ### Importance sampling

# Finally, we have a good estimate for the parameter to use with importance
# sampling. Gutzwiller.jl provides `AnsatzSampling`, which is similar to
# `GutzwillerSampling` from Rimu, but can be used with different ansatze.

p = grad_result.param[end]
G = AnsatzSampling(H, ansatz, p)

# This can now be used with FCIQMC. Let's compare the importance sampled time series to a
# non-importance sampled one.

using Random; Random.seed!(1337) #hide

prob_standard = ProjectorMonteCarloProblem(H; target_walkers=15, last_step=2000)
sim_standard = solve(prob_standard)
shift_estimator(sim_standard; skip=1000)

#

prob_sampled = ProjectorMonteCarloProblem(G; target_walkers=15, last_step=2000)
sim_sampled = solve(prob_sampled)
shift_estimator(sim_sampled; skip=1000)

# Note that the lower energy estimate in the sampled case is probably due to a reduced
# population control bias. The effect importance sampling has on the statistic can be more
# dramatic for larger systems and beter choices of anstaze.
