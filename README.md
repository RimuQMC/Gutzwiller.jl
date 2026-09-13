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
GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardReal1D{Float64, BoseFS{10, 10, BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0))
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
LocalEnergyEvaluator(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0), GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardReal1D{Float64, BoseFS{10, 10, BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0)))
````

which can be used to evaulate the value of the Rayleigh quotient (or its gradient) for
given parameters. In the case of the Gutzwiller ansatz, there is only one parameter.

````julia
le([1.0])
````

````
-7.825819465045137
````

Like before, we can use `val_and_grad` to also evaluate its gradient.

````julia
val_and_grad(le, [1.0])
````

````
(-7.825819465045137, [10.614147776160687])
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
    |f(x) - f(x')|         = 3.98e-13 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 3.06e-14 ≰ 0.0e+00
    |g(x)|                 = 1.61e-12 ≤ 1.0e-08

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
([0.3331889106038858], -13.005208186381465)
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
  walkers:      10
  samples:      1000
  local energy: -7.4863 ± 0.12108
````

````julia
@time kinetic_vqmc(H, ansatz, p0; steps=1e5)
````

````
KineticVQMCResult
  walkers:      10
  samples:      1000000
  local energy: -7.8217 ± 0.005304
````

````julia
@time kinetic_vqmc(H, ansatz, p0; steps=1e7)
````

````
KineticVQMCResult
  walkers:      10
  samples:      100000000
  local energy: -7.826 ± 0.00053919
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
  GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardReal1D{Float64, BoseFS{10, 10, BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0));
  steps=1000,
  walkers=10,
)
````

````julia
qmc([1.0]), le([1.0])
````

````
(-7.737078172189174, -7.825819465045137)
````

Because the output of this procedure is noisy, optimizing it with Optim.jl will not work.
However, we can use a stochastic gradient descent (in this case
[AMSGrad](https://paperswithcode.com/method/amsgrad)).

````julia
grad_result = amsgrad(qmc, [1.0])
````

````
GradientDescentResult
  iterations: 101
  converged: false (iterations)
  last value: -13.056634424022656 ± 0.04150448541507224
  last params: [0.33530117393649445]
````

While `amsgrad` attempts to determine if the optimization converged, it will generally not
detect convergence due to the noise in the QMC evaluation. The best way to determine
convergence is to plot the results. `grad_result` can be converted to a `DataFrame`.

````julia
grad_df = DataFrame(grad_result)
````

````
101×11 DataFrame
 Row │ α        β1       β2       iter   param       value      error      gradient       first_moment  second_moment  param_delta
     │ Float64  Float64  Float64  Int64  SArray…     Float64    Float64    SArray…        SArray…       SArray…        SArray…
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │    0.01      0.1     0.01      1  [1.0]        -7.84677  0.0453831  [10.8873]      [10.8854]     [1.18534]      [-0.0999824]
   2 │    0.01      0.1     0.01      2  [0.900018]   -8.94825  0.0454377  [10.609]       [10.8578]     [2.299]        [-0.0716096]
   3 │    0.01      0.1     0.01      3  [0.828408]   -9.58472  0.0455468  [9.83181]      [10.7552]     [3.24265]      [-0.0597265]
   4 │    0.01      0.1     0.01      4  [0.768681]  -10.2204   0.0470805  [9.5884]       [10.6385]     [4.1296]       [-0.0523512]
   5 │    0.01      0.1     0.01      5  [0.71633]   -10.7282   0.0473182  [9.33389]      [10.508]      [4.95952]      [-0.0471848]
   6 │    0.01      0.1     0.01      6  [0.669146]  -11.1963   0.0460688  [8.94124]      [10.3514]     [5.70938]      [-0.0433214]
   7 │    0.01      0.1     0.01      7  [0.625824]  -11.5181   0.0448215  [8.22758]      [10.139]      [6.32922]      [-0.0403013]
   8 │    0.01      0.1     0.01      8  [0.585523]  -11.852    0.0449426  [8.01151]      [9.92623]     [6.90777]      [-0.0377673]
   9 │    0.01      0.1     0.01      9  [0.547756]  -12.1407   0.0438294  [7.19498]      [9.65311]     [7.35637]      [-0.0355906]
  10 │    0.01      0.1     0.01     10  [0.512165]  -12.3369   0.0400098  [6.63062]      [9.35086]     [7.72246]      [-0.0336491]
  11 │    0.01      0.1     0.01     11  [0.478516]  -12.5376   0.0421786  [5.73274]      [8.98905]     [7.97388]      [-0.0318331]
  12 │    0.01      0.1     0.01     12  [0.446683]  -12.6732   0.04176    [4.46702]      [8.53684]     [8.09368]      [-0.0300071]
  13 │    0.01      0.1     0.01     13  [0.416676]  -12.8063   0.0384845  [3.72637]      [8.0558]      [8.1516]       [-0.0282155]
  14 │    0.01      0.1     0.01     14  [0.38846]   -12.9431   0.0412046  [2.93535]      [7.54375]     [8.15625]      [-0.0264145]
  15 │    0.01      0.1     0.01     15  [0.362046]  -13.0218   0.0400677  [1.86491]      [6.97587]     [8.15625]      [-0.024426]
  16 │    0.01      0.1     0.01     16  [0.33762]   -13.0268   0.0409747  [0.363188]     [6.3146]      [8.15625]      [-0.0221106]
  17 │    0.01      0.1     0.01     17  [0.315509]  -13.0504   0.0413928  [-0.951417]    [5.588]       [8.15625]      [-0.0195664]
  18 │    0.01      0.1     0.01     18  [0.295943]  -13.0103   0.0484318  [-2.49057]     [4.78014]     [8.15625]      [-0.0167377]
  19 │    0.01      0.1     0.01     19  [0.279205]  -12.9879   0.0441251  [-3.16686]     [3.98544]     [8.17498]      [-0.013939]
  20 │    0.01      0.1     0.01     20  [0.265266]  -12.9616   0.0424887  [-4.16739]     [3.17016]     [8.2669]       [-0.0110258]
  21 │    0.01      0.1     0.01     21  [0.25424]   -12.8003   0.050197   [-5.68721]     [2.28442]     [8.50767]      [-0.00783197]
  22 │    0.01      0.1     0.01     22  [0.246408]  -12.7291   0.0562879  [-6.88139]     [1.36784]     [8.89613]      [-0.00458601]
  23 │    0.01      0.1     0.01     23  [0.241822]  -12.7565   0.0512231  [-6.50647]     [0.58041]     [9.23051]      [-0.00191039]
  24 │    0.01      0.1     0.01     24  [0.239912]  -12.7255   0.0547702  [-6.91339]     [-0.16897]    [9.61615]      [0.000544891]
  25 │    0.01      0.1     0.01     25  [0.240457]  -12.6708   0.0539385  [-7.2258]      [-0.874653]   [10.0421]      [0.00276009]
  26 │    0.01      0.1     0.01     26  [0.243217]  -12.6267   0.0596866  [-6.98245]     [-1.48543]    [10.4292]      [0.00459967]
  27 │    0.01      0.1     0.01     27  [0.247816]  -12.7398   0.0545251  [-6.55496]     [-1.99239]    [10.7546]      [0.00607541]
  28 │    0.01      0.1     0.01     28  [0.253892]  -12.8572   0.0496218  [-5.30545]     [-2.32369]    [10.9286]      [0.00702906]
  29 │    0.01      0.1     0.01     29  [0.260921]  -12.7025   0.0626577  [-5.96273]     [-2.6876]     [11.1748]      [0.00803977]
  30 │    0.01      0.1     0.01     30  [0.268961]  -12.9296   0.0416227  [-3.99749]     [-2.81859]    [11.2229]      [0.00841355]
  31 │    0.01      0.1     0.01     31  [0.277374]  -12.9355   0.0453485  [-3.60385]     [-2.89711]    [11.2405]      [0.00864116]
  32 │    0.01      0.1     0.01     32  [0.286015]  -12.9939   0.0450852  [-2.92423]     [-2.89982]    [11.2405]      [0.00864925]
  33 │    0.01      0.1     0.01     33  [0.294665]  -13.0701   0.0420815  [-1.97424]     [-2.80727]    [11.2405]      [0.00837318]
  34 │    0.01      0.1     0.01     34  [0.303038]  -12.9648   0.0518133  [-1.88922]     [-2.71546]    [11.2405]      [0.00809936]
  35 │    0.01      0.1     0.01     35  [0.311137]  -13.0755   0.0355712  [-0.974903]    [-2.54141]    [11.2405]      [0.00758021]
  36 │    0.01      0.1     0.01     36  [0.318717]  -13.0263   0.0449461  [-0.350612]    [-2.32233]    [11.2405]      [0.00692676]
  37 │    0.01      0.1     0.01     37  [0.325644]  -12.9106   0.0436311  [-0.665099]    [-2.1566]     [11.2405]      [0.00643246]
  38 │    0.01      0.1     0.01     38  [0.332077]  -12.918    0.0455451  [-0.0290947]   [-1.94385]    [11.2405]      [0.00579789]
  39 │    0.01      0.1     0.01     39  [0.337874]  -13.0529   0.0405582  [0.929811]     [-1.65649]    [11.2405]      [0.00494077]
  40 │    0.01      0.1     0.01     40  [0.342815]  -13.0103   0.043652   [0.358596]     [-1.45498]    [11.2405]      [0.00433974]
  41 │    0.01      0.1     0.01     41  [0.347155]  -13.0623   0.0405893  [1.0418]       [-1.2053]     [11.2405]      [0.00359503]
  42 │    0.01      0.1     0.01     42  [0.35075]   -12.9576   0.0434175  [0.486975]     [-1.03607]    [11.2405]      [0.00309028]
  43 │    0.01      0.1     0.01     43  [0.35384]   -13.0169   0.039146   [0.979955]     [-0.83447]    [11.2405]      [0.00248896]
  44 │    0.01      0.1     0.01     44  [0.356329]  -12.944    0.0400163  [1.24146]      [-0.626877]   [11.2405]      [0.00186977]
  45 │    0.01      0.1     0.01     45  [0.358199]  -12.9201   0.0411648  [1.10555]      [-0.453634]   [11.2405]      [0.00135305]
  46 │    0.01      0.1     0.01     46  [0.359552]  -12.9848   0.0408427  [1.23679]      [-0.284592]   [11.2405]      [0.000848848]
  47 │    0.01      0.1     0.01     47  [0.360401]  -13.0529   0.0407539  [1.45605]      [-0.110528]   [11.2405]      [0.000329671]
  48 │    0.01      0.1     0.01     48  [0.360731]  -12.9483   0.0448997  [1.51407]      [0.051931]    [11.2405]      [-0.000154894]
  49 │    0.01      0.1     0.01     49  [0.360576]  -12.9413   0.0430997  [1.31569]      [0.178307]    [11.2405]      [-0.000531833]
  50 │    0.01      0.1     0.01     50  [0.360044]  -12.9942   0.0404768  [1.49975]      [0.310452]    [11.2405]      [-0.000925979]
  51 │    0.01      0.1     0.01     51  [0.359118]  -13.0083   0.0381456  [1.67559]      [0.446965]    [11.2405]      [-0.00133316]
  52 │    0.01      0.1     0.01     52  [0.357785]  -12.9988   0.046065   [1.35913]      [0.538182]    [11.2405]      [-0.00160523]
  53 │    0.01      0.1     0.01     53  [0.356179]  -12.9974   0.0393125  [1.26897]      [0.611261]    [11.2405]      [-0.0018232]
  54 │    0.01      0.1     0.01     54  [0.354356]  -13.0636   0.0405254  [1.64914]      [0.715049]    [11.2405]      [-0.00213276]
  55 │    0.01      0.1     0.01     55  [0.352224]  -12.9768   0.0446558  [0.862934]     [0.729837]    [11.2405]      [-0.00217687]
  56 │    0.01      0.1     0.01     56  [0.350047]  -12.9174   0.0452137  [0.224522]     [0.679306]    [11.2405]      [-0.00202615]
  57 │    0.01      0.1     0.01     57  [0.348021]  -12.9572   0.0445192  [0.66707]      [0.678082]    [11.2405]      [-0.0020225]
  58 │    0.01      0.1     0.01     58  [0.345998]  -13.0061   0.0418411  [0.796596]     [0.689934]    [11.2405]      [-0.00205785]
  59 │    0.01      0.1     0.01     59  [0.34394]   -12.9943   0.0390729  [0.802998]     [0.70124]     [11.2405]      [-0.00209158]
  60 │    0.01      0.1     0.01     60  [0.341849]  -12.997    0.046184   [0.0799629]    [0.639112]    [11.2405]      [-0.00190627]
  61 │    0.01      0.1     0.01     61  [0.339942]  -12.9886   0.0417714  [0.47894]      [0.623095]    [11.2405]      [-0.00185849]
  62 │    0.01      0.1     0.01     62  [0.338084]  -13.0688   0.041897   [0.509839]     [0.61177]     [11.2405]      [-0.00182471]
  63 │    0.01      0.1     0.01     63  [0.336259]  -13.004    0.0428121  [-0.0091042]   [0.549682]    [11.2405]      [-0.00163953]
  64 │    0.01      0.1     0.01     64  [0.33462]   -13.0173   0.0441219  [0.204541]     [0.515168]    [11.2405]      [-0.00153658]
  65 │    0.01      0.1     0.01     65  [0.333083]  -13.106    0.03642    [0.445926]     [0.508244]    [11.2405]      [-0.00151593]
  66 │    0.01      0.1     0.01     66  [0.331567]  -12.9908   0.0491206  [-0.309975]    [0.426422]    [11.2405]      [-0.00127188]
  67 │    0.01      0.1     0.01     67  [0.330295]  -12.9184   0.0458485  [-0.058245]    [0.377955]    [11.2405]      [-0.00112732]
  68 │    0.01      0.1     0.01     68  [0.329168]  -13.0959   0.0389771  [0.244675]     [0.364627]    [11.2405]      [-0.00108757]
  69 │    0.01      0.1     0.01     69  [0.32808]   -13.0083   0.0424406  [-0.164715]    [0.311693]    [11.2405]      [-0.000929681]
  70 │    0.01      0.1     0.01     70  [0.327151]  -12.9683   0.0440395  [-0.246347]    [0.255889]    [11.2405]      [-0.000763236]
  71 │    0.01      0.1     0.01     71  [0.326387]  -13.0235   0.0406738  [-0.392452]    [0.191055]    [11.2405]      [-0.000569856]
  72 │    0.01      0.1     0.01     72  [0.325818]  -13.0777   0.0404443  [-0.219194]    [0.15003]     [11.2405]      [-0.000447492]
  73 │    0.01      0.1     0.01     73  [0.32537]   -12.9618   0.0446183  [-0.404058]    [0.0946211]   [11.2405]      [-0.000282225]
  74 │    0.01      0.1     0.01     74  [0.325088]  -12.9788   0.041135   [-0.278374]    [0.0573216]   [11.2405]      [-0.000170972]
  75 │    0.01      0.1     0.01     75  [0.324917]  -13.0188   0.0407611  [-0.176731]    [0.0339164]   [11.2405]      [-0.000101162]
  76 │    0.01      0.1     0.01     76  [0.324816]  -13.0506   0.0378133  [-0.150099]    [0.0155148]   [11.2405]      [-4.62757e-5]
  77 │    0.01      0.1     0.01     77  [0.324769]  -13.062    0.0390751  [-0.0980721]   [0.00415611]  [11.2405]      [-1.23963e-5]
  78 │    0.01      0.1     0.01     78  [0.324757]  -13.0261   0.0420914  [-0.346088]    [-0.0308683]  [11.2405]      [9.20702e-5]
  79 │    0.01      0.1     0.01     79  [0.324849]  -12.9766   0.0450654  [-0.434911]    [-0.0712725]  [11.2405]      [0.000212583]
  80 │    0.01      0.1     0.01     80  [0.325062]  -13.059    0.038719   [-0.305888]    [-0.0947341]  [11.2405]      [0.000282562]
  81 │    0.01      0.1     0.01     81  [0.325344]  -12.9881   0.0431456  [-0.365783]    [-0.121839]   [11.2405]      [0.000363407]
  82 │    0.01      0.1     0.01     82  [0.325708]  -13.1236   0.0402063  [0.27908]      [-0.0817471]  [11.2405]      [0.000243826]
  83 │    0.01      0.1     0.01     83  [0.325951]  -13.0694   0.0424114  [-0.30733]     [-0.104305]   [11.2405]      [0.00031111]
  84 │    0.01      0.1     0.01     84  [0.326263]  -12.9757   0.0424166  [-0.572783]    [-0.151153]   [11.2405]      [0.000450842]
  85 │    0.01      0.1     0.01     85  [0.326713]  -12.9366   0.0482267  [-0.76629]     [-0.212667]   [11.2405]      [0.000634318]
  86 │    0.01      0.1     0.01     86  [0.327348]  -12.9622   0.0414355  [-0.322291]    [-0.223629]   [11.2405]      [0.000667015]
  87 │    0.01      0.1     0.01     87  [0.328015]  -12.9666   0.0439625  [-0.0775585]   [-0.209022]   [11.2405]      [0.000623447]
  88 │    0.01      0.1     0.01     88  [0.328638]  -13.0293   0.0408462  [-0.156193]    [-0.203739]   [11.2405]      [0.00060769]
  89 │    0.01      0.1     0.01     89  [0.329246]  -13.0206   0.0458504  [0.0261816]    [-0.180747]   [11.2405]      [0.000539111]
  90 │    0.01      0.1     0.01     90  [0.329785]  -12.937    0.0529258  [-0.859567]    [-0.248629]   [11.2405]      [0.000741582]
  91 │    0.01      0.1     0.01     91  [0.330527]  -13.0206   0.0399218  [-0.00129799]  [-0.223896]   [11.2405]      [0.000667811]
  92 │    0.01      0.1     0.01     92  [0.331194]  -13.0403   0.0428934  [-0.0914775]   [-0.210654]   [11.2405]      [0.000628315]
  93 │    0.01      0.1     0.01     93  [0.331823]  -12.9434   0.0440219  [0.129138]     [-0.176675]   [11.2405]      [0.000526965]
  94 │    0.01      0.1     0.01     94  [0.33235]   -12.9467   0.0521169  [-0.295262]    [-0.188534]   [11.2405]      [0.000562336]
  95 │    0.01      0.1     0.01     95  [0.332912]  -13.0129   0.0448723  [-0.172248]    [-0.186905]   [11.2405]      [0.000557478]
  96 │    0.01      0.1     0.01     96  [0.333469]  -13.0068   0.0426976  [0.189407]     [-0.149274]   [11.2405]      [0.000445236]
  97 │    0.01      0.1     0.01     97  [0.333915]  -12.9853   0.045917   [-0.124807]    [-0.146827]   [11.2405]      [0.000437939]
  98 │    0.01      0.1     0.01     98  [0.334353]  -13.0323   0.0396698  [0.147685]     [-0.117376]   [11.2405]      [0.000350095]
  99 │    0.01      0.1     0.01     99  [0.334703]  -13.0529   0.0388572  [0.36878]      [-0.0687604]  [11.2405]      [0.00020509]
 100 │    0.01      0.1     0.01    100  [0.334908]  -12.8189   0.0499152  [-0.699951]    [-0.131879]   [11.2405]      [0.000393354]
 101 │    0.01      0.1     0.01    101  [0.335301]  -13.0566   0.0415045  [0.598479]     [-0.0588435]  [11.2405]      [0.000175512]
````

To plot it, we can either work with the `DataFrame` or access the fields directly

````julia
begin
    fig = Figure()
    ax1 = Axis(fig[1, 1]; xlabel=L"i", ylabel=L"E_v")
    ax2 = Axis(fig[2, 1]; xlabel=L"i", ylabel=L"p")

    lines!(ax1, grad_df.iter, grad_df.value)
    lines!(ax2, first.(grad_result.param))
    fig
end
````
![](README-41.png)

We see from the plot that the value of the energy is fluctiating around what appears to be
the minimum. While the parameter estimate here is probably good enough for importance
sampling, we can refine the result by creating a new `KineticVQMC` structure with
increased samples and use it to refine the result. Here, we can pass the previous result
`grad_result` in place of the initial parameters, which will continue the computation
where the previous one left off. Alternatively, this can be achieved by passing the `first_moment_init` and `second_moment_init` arguments to `amsgrad`.

````julia
qmc2 = KineticVQMC(H, ansatz; samples=1e6)
grad_result2 = amsgrad(qmc2, grad_result)
````

````
GradientDescentResult
  iterations: 101
  converged: false (iterations)
  last value: -12.993322382922992 ± 0.005561698350727309
  last params: [0.3332521705284508]
````

Now, let's plot the refined result next to the minimum found by Optim.jl

````julia
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
````
![](README-45.png)

### Importance sampling

Finally, we have a good estimate for the parameter to use with importance
sampling. Gutzwiller.jl provides `AnsatzSampling`, which is similar to
`GutzwillerSampling` from Rimu, but can be used with different ansatze.

````julia
p = grad_result.param[end]
G = AnsatzSampling(H, ansatz, p)
````

````
AnsatzSampling{false, Float64, 1, GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardReal1D{Float64, BoseFS{10, 10, BitString{19, 1, UInt32}}, 2.0, 1.0}}, HubbardReal1D{Float64, BoseFS{10, 10, BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0), GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardReal1D{Float64, BoseFS{10, 10, BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0)), [0.33530117393649445])
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
  mean = -12.28 ± 0.64
  with uncertainty of ± 0.08265153828812362
  from 31 blocks after 5 transformations (k = 6).

````

````julia
prob_sampled = ProjectorMonteCarloProblem(G; target_walkers=15, last_step=2000)
sim_sampled = solve(prob_sampled)
shift_estimator(sim_sampled; skip=1000)
````

````
BlockingResult{Float64}
  mean = -12.61 ± 0.27
  with uncertainty of ± 0.02426792029257624
  from 62 blocks after 4 transformations (k = 5).

````

Note that the lower energy estimate in the sampled case is probably due to a reduced
population control bias. The effect importance sampling has on the statistic can be more
dramatic for larger systems and beter choices of anstaze.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

