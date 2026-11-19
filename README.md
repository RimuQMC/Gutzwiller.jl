# Gutzwiller

[![Coverage Status](https://coveralls.io/repos/github/mtsch/Gutzwiller.jl/badge.svg?branch=master)](https://coveralls.io/github/mtsch/Gutzwiller.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://rimuqmc.github.io/Gutzwiller.jl/)

_importance sampling and variational Monte Carlo for
[Rimu.jl](https://github.com/joachimbrand/Rimu.jl)_

## Installation

Gutzwiller.jl is not yet registered. To install it, run

```julia
import Pkg; Pkg.add("https://github.com/mtsch/Gutzwiller.jl")
```

## Usage guide

````julia
using Rimu
using Gutzwiller
using CairoMakie
using LaTeXStrings
````

First, we set up a starting address and a Hamiltonian

````julia
addr = near_uniform(BoseFS{10,10})
H = HubbardReal1D(addr; u=2.0)
````

````
HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0)
````

In this example, we'll set up a Gutzwiller ansatz to importance-sample the Hamiltonian.

````julia
ansatz = GutzwillerAnsatz(H)
````

````
Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0))
````

An ansatz is a struct that given a set of parameters and an address, produces the value
it would have if it was a vector.

````julia
ansatz(addr, [1.0])
````

````
1.0
````

In addition, the function `val_and_grad` can be used to compute both the value and its
gradient with respect to the parameters.

````julia
val_and_grad(ansatz, addr, [1.0])
````

````
(1.0, [-0.0])
````

### Deterministic optimization

For effective importance sampling, we want the ansatz to be as good of an approximation to
the ground state of the Hamiltonian as possible. As the value of the Rayleigh quotient of
a given ansatz is always larger than the Hamiltonian's ground state energy, we can use
an optimization algorithm to find the paramters that minimize its energy.

When the basis of the Hamiltonian is small enough to fit into memory, it's best to use the
`LocalEnergyEvaluator`

````julia
le = LocalEnergyEvaluator(H, ansatz)
````

````
LocalEnergyEvaluator(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0), Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0)))
````

which can be used to evaulate the value of the Rayleigh quotient (or its gradient) for
given parameters. In the case of the Gutzwiller ansatz, there is only one parameter.

````julia
le([1.0])
````

````
-7.8258194650481325
````

Like before, we can use `val_and_grad` to also evaluate its gradient.

````julia
val_and_grad(le, [1.0])
````

````
(-7.8258194650481325, [10.614147776149276])
````

Now, let's plot the energy landscape for this particular case

````julia
begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"p", ylabel=L"E")
    ps = range(0, 2; length=100)
    Es = [le([p]) for p in ps]
    lines!(ax, ps, Es)
    fig
end
````
![](README-21.png)

To find the minimum, pass `le` to `optimize` from Optim.jl

````julia
using Optim, NLSolversBase

opt_nelder = optimize(le, [1.0])
````

````
 * Status: success

 * Candidate solution
    Final objective value:     -1.300521e+01

 * Found with
    Algorithm:     Nelder-Mead

 * Convergence measures
    √(Σ(yᵢ-ȳ)²)/n ≤ 1.0e-08

 * Work counters
    Seconds run:   0  (vs limit Inf)
    Iterations:    11
    f(x) calls:    25

````

To take advantage of the gradients, wrap the evaluator in `only_fg!`. This will
usually reduce the number of steps needed to reach the minimum.

````julia
opt_lbgfs = optimize(only_fg!(le), [1.0])
````

````
 * Status: success

 * Candidate solution
    Final objective value:     -1.300521e+01

 * Found with
    Algorithm:     L-BFGS

 * Convergence measures
    |x - x'|               = 1.01e-08 ≰ 0.0e+00
    |x - x'|/|x'|          = 3.03e-08 ≰ 0.0e+00
    |f(x) - f(x')|         = 3.23e-12 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 2.48e-13 ≰ 0.0e+00
    |g(x)|                 = 3.82e-11 ≤ 1.0e-08

 * Work counters
    Seconds run:   1  (vs limit Inf)
    Iterations:    8
    f(x) calls:    14
    ∇f(x) calls:   14
    ∇f(x)ᵀv calls: 0

````

We can inspect the parameters and the value at the minimum as

````julia
opt_lbgfs.minimizer, opt_lbgfs.minimum
````

````
([0.333188910603056], -13.005208186381019)
````

### Variational quantum Monte Carlo

When the Hamiltonian is too large to store its full basis in memory, we can use
variational QMC to sample addresses from the Hilbert space and evaluate their energy
at the same time. An important paramter we have tune is the number `steps`. More steps
will give us a better approximation of the energy, but take longer to evaluate.
Not taking enough samples can also result in producing a biased result.
Consider the following.

````julia
p0 = [1.0]
@time kinetic_vqmc(H, ansatz, p0; steps=1e2)
````

````
KineticVQMCResult
  walkers:      8
  samples:      800
  local energy: -8.0264 ± 0.15589
````

````julia
@time kinetic_vqmc(H, ansatz, p0; steps=1e5)
````

````
KineticVQMCResult
  walkers:      8
  samples:      800000
  local energy: -7.8274 ± 0.0057852
````

````julia
@time kinetic_vqmc(H, ansatz, p0; steps=1e7)
````

````
KineticVQMCResult
  walkers:      8
  samples:      80000000
  local energy: -7.8259 ± 0.00060239
````

For this simple example, `1e2` steps gives an energy that is significantly higher, while
`1e7` takes too long. `1e5` seems to work well enough. For more convenient evaluation, we
wrap VQMC into a struct that behaves much like the `LocalEnergyEvaluator`.

````julia
qmc = KineticVQMC(H, ansatz; samples=1e4)
````

````
KineticVQMC(
  HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0),
  Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0));
  steps=1250,
  walkers=8,
)
````

````julia
qmc([1.0]), le([1.0])
````

````
(-7.924441077227302, -7.8258194650481325)
````

Because the output of this procedure is noisy, optimizing it with Optim.jl will not work.
Instead, we use
[stochastic reconfiguration](https://doi.org/10.1103/PhysRevB.64.024512), which
estimates the gradient together with the metric tensor of the ansatz from the same
samples, and uses the metric to choose well-scaled steps.

````julia
sr_result = stochastic_reconfiguration(qmc, [1.0])
````

````
StochasticReconfigurationResult
  iterations: 100
  converged: false (iterations)
  last value: -13.041925051744894 ± 0.04576468320063123
  last params: [0.3323055701870732]
````

While `stochastic_reconfiguration` attempts to determine if the optimization converged,
it will generally not detect convergence due to the noise in the QMC evaluation. The best
way to determine convergence is to plot the results. `sr_result` can be converted to a
`DataFrame`.

````julia
sr_df = DataFrame(sr_result)
````

````
100×8 DataFrame
 Row │ τ        shift    iter   param       value      error      gradient      param_delta
     │ Float64  Float64  Int64  SArray…     Float64    Float64    SArray…       SArray…
─────┼─────────────────────────────────────────────────────────────────────────────────────────
   1 │    0.05    0.001      1  [1.0]        -7.85657  0.0434279  [10.7155]     [-0.146187]
   2 │    0.05    0.001      2  [0.853813]   -9.34629  0.0460339  [10.0174]     [-0.11686]
   3 │    0.05    0.001      3  [0.736952]  -10.5144   0.0487416  [9.60735]     [-0.0930245]
   4 │    0.05    0.001      4  [0.643928]  -11.322    0.0457453  [8.19707]     [-0.073792]
   5 │    0.05    0.001      5  [0.570136]  -11.8912   0.0440621  [7.07406]     [-0.0561831]
   6 │    0.05    0.001      6  [0.513953]  -12.3305   0.0404791  [6.35037]     [-0.0467806]
   7 │    0.05    0.001      7  [0.467172]  -12.6606   0.0371068  [5.64822]     [-0.0368077]
   8 │    0.05    0.001      8  [0.430364]  -12.7751   0.035998   [4.12588]     [-0.0256881]
   9 │    0.05    0.001      9  [0.404676]  -12.8723   0.0390116  [3.39581]     [-0.0203621]
  10 │    0.05    0.001     10  [0.384314]  -12.9556   0.0403141  [2.43974]     [-0.0132029]
  11 │    0.05    0.001     11  [0.371111]  -12.9706   0.0416357  [2.06728]     [-0.0109366]
  12 │    0.05    0.001     12  [0.360175]  -13.028    0.0419965  [1.57581]     [-0.00788942]
  13 │    0.05    0.001     13  [0.352285]  -12.8712   0.0491274  [0.780374]    [-0.00373916]
  14 │    0.05    0.001     14  [0.348546]  -13.1154   0.0385139  [1.14891]     [-0.00583624]
  15 │    0.05    0.001     15  [0.34271]   -13.0959   0.0449044  [0.721302]    [-0.00356675]
  16 │    0.05    0.001     16  [0.339143]  -12.9982   0.0483236  [0.0804455]   [-0.000374074]
  17 │    0.05    0.001     17  [0.338769]  -13.0659   0.0413554  [0.636381]    [-0.0030441]
  18 │    0.05    0.001     18  [0.335725]  -13.0835   0.041656   [0.364641]    [-0.00177974]
  19 │    0.05    0.001     19  [0.333945]  -13.006    0.0489358  [0.147016]    [-0.000670753]
  20 │    0.05    0.001     20  [0.333274]  -12.9666   0.0480532  [0.0121062]   [-5.70243e-5]
  21 │    0.05    0.001     21  [0.333217]  -13.0209   0.0454352  [0.0832993]   [-0.000390636]
  22 │    0.05    0.001     22  [0.332827]  -12.8709   0.0451922  [-0.405568]   [0.00186569]
  23 │    0.05    0.001     23  [0.334693]  -12.9043   0.0473498  [-0.14547]    [0.000690806]
  24 │    0.05    0.001     24  [0.335383]  -12.9439   0.0461484  [0.0416374]   [-0.000183212]
  25 │    0.05    0.001     25  [0.3352]    -12.9858   0.0481715  [-0.134268]   [0.000651658]
  26 │    0.05    0.001     26  [0.335852]  -12.9721   0.047284   [0.187519]    [-0.000842392]
  27 │    0.05    0.001     27  [0.335009]  -13.076    0.0472023  [0.463232]    [-0.00207834]
  28 │    0.05    0.001     28  [0.332931]  -12.9667   0.0459281  [0.0507581]   [-0.000233544]
  29 │    0.05    0.001     29  [0.332697]  -13.0504   0.0423257  [-0.0234475]  [0.000110057]
  30 │    0.05    0.001     30  [0.332808]  -12.9835   0.0462339  [-0.16497]    [0.000752076]
  31 │    0.05    0.001     31  [0.33356]   -12.9739   0.0401462  [-0.0426622]  [0.000200474]
  32 │    0.05    0.001     32  [0.33376]   -13.0434   0.0403968  [-0.018554]   [8.99364e-5]
  33 │    0.05    0.001     33  [0.33385]   -13.0461   0.0414121  [0.524116]    [-0.00243428]
  34 │    0.05    0.001     34  [0.331416]  -13.0182   0.043495   [-0.0143906]  [6.78225e-5]
  35 │    0.05    0.001     35  [0.331484]  -13.0242   0.0403021  [0.242507]    [-0.00114277]
  36 │    0.05    0.001     36  [0.330341]  -12.944    0.0459788  [-0.361046]   [0.00165316]
  37 │    0.05    0.001     37  [0.331994]  -12.9526   0.0463102  [0.152834]    [-0.000669217]
  38 │    0.05    0.001     38  [0.331325]  -13.0266   0.0439471  [-0.0509813]  [0.000243292]
  39 │    0.05    0.001     39  [0.331568]  -13.0517   0.0391991  [-0.133537]   [0.000642579]
  40 │    0.05    0.001     40  [0.332211]  -13.0925   0.0425985  [0.195193]    [-0.000915944]
  41 │    0.05    0.001     41  [0.331295]  -12.8993   0.0519902  [-0.611822]   [0.00287066]
  42 │    0.05    0.001     42  [0.334165]  -12.9133   0.0487314  [-0.482872]   [0.00211918]
  43 │    0.05    0.001     43  [0.336285]  -13.0634   0.0430115  [0.287487]    [-0.00139289]
  44 │    0.05    0.001     44  [0.334892]  -13.0741   0.0409595  [-0.0562891]  [0.000265956]
  45 │    0.05    0.001     45  [0.335158]  -13.0035   0.0410119  [0.433998]    [-0.00204551]
  46 │    0.05    0.001     46  [0.333112]  -12.9071   0.0463129  [-0.46755]    [0.0021516]
  47 │    0.05    0.001     47  [0.335264]  -12.9747   0.0493409  [-0.150479]   [0.000692261]
  48 │    0.05    0.001     48  [0.335956]  -12.9202   0.0417019  [-0.109201]   [0.000492804]
  49 │    0.05    0.001     49  [0.336449]  -13.0593   0.0432107  [0.182069]    [-0.00088103]
  50 │    0.05    0.001     50  [0.335568]  -13.0349   0.0433432  [0.231816]    [-0.00112909]
  51 │    0.05    0.001     51  [0.334439]  -13.0658   0.0447829  [0.341363]    [-0.00169191]
  52 │    0.05    0.001     52  [0.332747]  -12.9892   0.0446974  [-0.0829899]  [0.000374868]
  53 │    0.05    0.001     53  [0.333122]  -13.0695   0.0454131  [0.141387]    [-0.000644019]
  54 │    0.05    0.001     54  [0.332478]  -13.0258   0.0437176  [0.340318]    [-0.00159888]
  55 │    0.05    0.001     55  [0.330879]  -12.992    0.0470331  [-0.142993]   [0.000644047]
  56 │    0.05    0.001     56  [0.331523]  -12.8808   0.0456378  [-0.35478]    [0.00171281]
  57 │    0.05    0.001     57  [0.333236]  -13.0216   0.0469584  [-0.194599]   [0.000921363]
  58 │    0.05    0.001     58  [0.334157]  -13.0818   0.0394243  [0.147927]    [-0.000725083]
  59 │    0.05    0.001     59  [0.333432]  -13.1385   0.0372785  [0.545984]    [-0.00268388]
  60 │    0.05    0.001     60  [0.330748]  -13.08     0.0455698  [0.0647565]   [-0.000305102]
  61 │    0.05    0.001     61  [0.330443]  -13.1166   0.0402273  [0.0721739]   [-0.000335638]
  62 │    0.05    0.001     62  [0.330107]  -12.9559   0.0476391  [-0.812619]   [0.00367717]
  63 │    0.05    0.001     63  [0.333784]  -13.008    0.0394519  [0.104127]    [-0.000487138]
  64 │    0.05    0.001     64  [0.333297]  -12.9632   0.0492774  [-0.0403466]  [0.000186268]
  65 │    0.05    0.001     65  [0.333483]  -13.002    0.0431065  [0.168264]    [-0.000795989]
  66 │    0.05    0.001     66  [0.332688]  -13.0443   0.0440521  [0.0234369]   [-0.000107504]
  67 │    0.05    0.001     67  [0.33258]   -12.9735   0.0456923  [-0.316815]   [0.00151832]
  68 │    0.05    0.001     68  [0.334098]  -13.0045   0.0433131  [0.171531]    [-0.000758503]
  69 │    0.05    0.001     69  [0.33334]   -12.9766   0.0492327  [-0.246994]   [0.00109617]
  70 │    0.05    0.001     70  [0.334436]  -13.0714   0.0376083  [0.249651]    [-0.00120465]
  71 │    0.05    0.001     71  [0.333231]  -13.007    0.0442752  [-0.0551142]  [0.000266625]
  72 │    0.05    0.001     72  [0.333498]  -12.9514   0.0504994  [-0.601341]   [0.00256492]
  73 │    0.05    0.001     73  [0.336063]  -13.0684   0.0391151  [0.187949]    [-0.000910217]
  74 │    0.05    0.001     74  [0.335153]  -12.9887   0.0424174  [0.0425696]   [-0.000203786]
  75 │    0.05    0.001     75  [0.334949]  -12.975    0.0443467  [-0.196549]   [0.00093281]
  76 │    0.05    0.001     76  [0.335882]  -12.9545   0.0427629  [-0.127678]   [0.000624077]
  77 │    0.05    0.001     77  [0.336506]  -13.0126   0.04556    [-0.344264]   [0.0015474]
  78 │    0.05    0.001     78  [0.338053]  -12.9722   0.0469301  [0.129926]    [-0.000611714]
  79 │    0.05    0.001     79  [0.337441]  -12.9375   0.0471612  [0.41484]     [-0.00189308]
  80 │    0.05    0.001     80  [0.335548]  -12.9936   0.047269   [-0.134143]   [0.000657735]
  81 │    0.05    0.001     81  [0.336206]  -12.9919   0.0515173  [-0.0317011]  [0.0001456]
  82 │    0.05    0.001     82  [0.336352]  -13.0253   0.0473106  [0.280508]    [-0.00134472]
  83 │    0.05    0.001     83  [0.335007]  -12.9908   0.0502984  [-0.241116]   [0.00107741]
  84 │    0.05    0.001     84  [0.336084]  -12.9595   0.0431311  [0.486888]    [-0.00223226]
  85 │    0.05    0.001     85  [0.333852]  -13.0478   0.0450905  [0.0562214]   [-0.000258422]
  86 │    0.05    0.001     86  [0.333594]  -12.9865   0.0402557  [0.186506]    [-0.000850703]
  87 │    0.05    0.001     87  [0.332743]  -12.9357   0.0436552  [0.255702]    [-0.00121905]
  88 │    0.05    0.001     88  [0.331524]  -13.0283   0.0415061  [0.23553]     [-0.00113332]
  89 │    0.05    0.001     89  [0.330391]  -12.9694   0.0417967  [-0.286653]   [0.00132358]
  90 │    0.05    0.001     90  [0.331714]  -12.9949   0.0397719  [0.0503362]   [-0.000234169]
  91 │    0.05    0.001     91  [0.33148]   -12.9809   0.0414356  [-0.0613847]  [0.000294828]
  92 │    0.05    0.001     92  [0.331775]  -12.9479   0.0484531  [-0.320632]   [0.00145363]
  93 │    0.05    0.001     93  [0.333228]  -12.9619   0.0445609  [-0.301436]   [0.00139636]
  94 │    0.05    0.001     94  [0.334625]  -12.9562   0.0465348  [0.0616585]   [-0.000290002]
  95 │    0.05    0.001     95  [0.334335]  -12.9536   0.0465484  [-0.19037]    [0.000912781]
  96 │    0.05    0.001     96  [0.335248]  -13.0097   0.04786    [-0.0841074]  [0.000400036]
  97 │    0.05    0.001     97  [0.335648]  -13.096    0.0392096  [0.654128]    [-0.0032262]
  98 │    0.05    0.001     98  [0.332421]  -12.9922   0.0442638  [0.0563668]   [-0.000262538]
  99 │    0.05    0.001     99  [0.332159]  -13.0379   0.0463634  [-0.0332675]  [0.00014665]
 100 │    0.05    0.001    100  [0.332306]  -13.0419   0.0457647  [0.0927356]   [-0.000430043]
````

To plot it, we can either work with the `DataFrame` or access the fields directly

````julia
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xlabel=L"i", ylabel=L"E_v")
    ax2 = Axis(fig[2, 1]; xlabel=L"i", ylabel=L"p")

    lines!(ax1, sr_df.iter, sr_df.value)
    lines!(ax2, first.(sr_result.param))
    fig
end
````
![](README-41.png)

We see from the plot that the value of the energy is fluctuating around what appears to be
the minimum. While the parameter estimate here is probably good enough for importance
sampling, we can refine the result by creating a new `KineticVQMC` structure with
increased samples and use it to refine the result. Here, we can pass the previous result
`sr_result` in place of the initial parameters, which will continue the computation
where the previous one left off.

````julia
qmc2 = KineticVQMC(H, ansatz; samples=1e6)
sr_result2 = stochastic_reconfiguration(qmc2, sr_result; maxiter=20)
````

````
StochasticReconfigurationResult
  iterations: 20
  converged: false (iterations)
  last value: -13.005077519828966 ± 0.005549205613770955
  last params: [0.33315687009262274]
````

Now, let's plot the refined result next to the minimum found by Optim.jl

````julia
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xlabel=L"i", ylabel=L"E_v")
    ax2 = Axis(fig[2, 1]; xlabel=L"i", ylabel=L"p")

    lines!(ax1, sr_result2.value)
    hlines!(ax1, [opt_lbgfs.minimum]; linestyle=:dot)
    lines!(ax2, first.(sr_result2.param))
    hlines!(ax2, opt_lbgfs.minimizer; linestyle=:dot)
    fig
end
````
![](README-45.png)

### Importance sampling

Finally, we have a good estimate for the parameter to use with importance
sampling. Gutzwiller.jl provides `AnsatzSampling`, which is similar to
`GutzwillerSampling` from Rimu, but can be used with different ansatze.

````julia
p = final_params(sr_result2)
G = AnsatzSampling(H, ansatz, p)
````

````
Gutzwiller.AnsatzSampling{false, Float64, 1, Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0), Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0)), [0.3330368347343056])
````

This can now be used with FCIQMC. Let's compare the importance sampled time series to a
non-importance sampled one.

````julia

prob_standard = ProjectorMonteCarloProblem(H; target_walkers=15, last_step=2000)
sim_standard = solve(prob_standard)
shift_estimator(sim_standard; skip=1000)
````

````
BlockingResult{Float64}
  mean = -11.08 ± 0.68
  with uncertainty of ± 0.08808725488030232
  from 31 blocks after 5 transformations (k = 6).

````

````julia
prob_sampled = ProjectorMonteCarloProblem(G; target_walkers=15, last_step=2000)
sim_sampled = solve(prob_sampled)
shift_estimator(sim_sampled; skip=1000)
````

````
BlockingResult{Float64}
  mean = -12.87 ± 0.36
  with uncertainty of ± 0.04598383300911905
  from 31 blocks after 5 transformations (k = 6).

````

Note that the lower energy estimate in the sampled case is probably due to a reduced
population control bias. The effect importance sampling has on the statistic can be more
dramatic for larger systems and beter choices of anstaze.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

