# Gutzwiller

[![Coverage Status](https://coveralls.io/repos/github/RimuQMC/Gutzwiller.jl/badge.svg?branch=master)](https://coveralls.io/github/RimuQMC/Gutzwiller.jl?branch=master)

_importance sampling and variational Monte Carlo for
[Rimu.jl](https://github.com/RimuQMC/Rimu.jl)_

## Installation

Gutzwiller.jl is not yet registered. To install it, run

```julia
import Pkg; Pkg.add(url="https://github.com/RimuQMC/Gutzwiller.jl")
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
-7.825819465047312
````

Like before, we can use `val_and_grad` to also evaluate its gradient.

````julia
val_and_grad(le, [1.0])
````

````
(-7.825819465047312, [10.614147776143358])
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
![](docs/README-21.png)

To find the minimum, pass `le` to `optimize` from Optim.jl

````julia
using Optim

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
    Seconds run:   1  (vs limit Inf)
    Iterations:    11
    f(x) calls:    25

````

To take advantage of the gradients, wrap the evaluator in `NLSolversBase.only_fg!`. This
will usually reduce the number of steps needed to reach the minimum.

````julia
opt_lbgfs = optimize(NLSolversBase.only_fg!(le), [1.0])
````

````
 * Status: success

 * Candidate solution
    Final objective value:     -1.300521e+01

 * Found with
    Algorithm:     L-BFGS

 * Convergence measures
    |x - x'|               = 4.19e-05 ≰ 0.0e+00
    |x - x'|/|x'|          = 1.26e-04 ≰ 0.0e+00
    |f(x) - f(x')|         = 5.04e-08 ≰ 0.0e+00
    |f(x) - f(x')|/|f(x')| = 3.87e-09 ≰ 0.0e+00
    |g(x)|                 = 4.80e-09 ≤ 1.0e-08

 * Work counters
    Seconds run:   1  (vs limit Inf)
    Iterations:    5
    f(x) calls:    16
    ∇f(x) calls:   16

````

We can inspect the parameters and the value at the minimum as

````julia
opt_lbgfs.minimizer, opt_lbgfs.minimum
````

````
([0.3331889106855026], -13.00520818638074)
````

### Variational quantum Monte Carlo

When the Hamiltonian is too large to store its full basis in memory, we can use
variational QMC to sample addresses from the Hilbert space and evaluate their energy
at the same time. An important parameter we have to tune is the number `steps`. More
steps will give us a better approximation of the energy, but take longer to evaluate.
Not taking enough samples can also result in producing a biased result.
Consider the following.

````julia
p0 = [1.0]
@time kinetic_vqmc(H, ansatz, p0; steps=1e2)
````

````
KineticVQMCResult
  walkers:      1
  samples:      100
  local energy: -7.3254 ± 0.33167
````

````julia
@time kinetic_vqmc(H, ansatz, p0; steps=1e5)
````

````
KineticVQMCResult
  walkers:      1
  samples:      100000
  local energy: -7.8369 ± 0.016555
````

````julia
@time kinetic_vqmc(H, ansatz, p0; steps=1e7)
````

````
KineticVQMCResult
  walkers:      1
  samples:      10000000
  local energy: -7.825 ± 0.0016978
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
  steps=10000,
  walkers=1,
)
````

````julia
qmc([1.0]), le([1.0])
````

````
(-7.828851494141587, -7.825819465047312)
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
  last value: -13.054226066948441
  last params: [0.33407952896097876]
````

While `amsgrad` attempts to determine if the optimization converged, it will generally not
detect convergence due to the noise in the QMC evaluation. The best way to determine
convergence is to plot the results. `grad_result` can be converted to a `DataFrame`.

````julia
grad_df = DataFrame(grad_result)
````

````
101×10 DataFrame
 Row │ α        β1       β2       iter   param       value      gradient       first_moment  second_moment  param_delta
     │ Float64  Float64  Float64  Int64  SArray…     Float64    SArray…        SArray…       SArray…        SArray…
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │    0.01      0.1     0.01      1  [1.0]        -7.86947  [10.6335]      [10.6726]     [1.13072]      [-0.100367]
   2 │    0.01      0.1     0.01      2  [0.899633]   -8.86381  [10.4539]      [10.6507]     [2.21224]      [-0.0716081]
   3 │    0.01      0.1     0.01      3  [0.828025]   -9.61802  [9.98253]      [10.5839]     [3.18663]      [-0.0592897]
   4 │    0.01      0.1     0.01      4  [0.768735]  -10.2245   [10.1458]      [10.5401]     [4.18414]      [-0.0515277]
   5 │    0.01      0.1     0.01      5  [0.717207]  -10.5959   [8.97968]      [10.384]      [4.94865]      [-0.0466792]
   6 │    0.01      0.1     0.01      6  [0.670528]  -11.1979   [9.08252]      [10.2539]     [5.72408]      [-0.0428584]
   7 │    0.01      0.1     0.01      7  [0.62767]   -11.5211   [8.55523]      [10.084]      [6.39876]      [-0.0398645]
   8 │    0.01      0.1     0.01      8  [0.587805]  -11.8079   [7.71817]      [9.84744]     [6.93047]      [-0.037406]
   9 │    0.01      0.1     0.01      9  [0.550399]  -11.9961   [6.90477]      [9.55317]     [7.33793]      [-0.0352664]
  10 │    0.01      0.1     0.01     10  [0.515133]  -12.3262   [6.60435]      [9.25829]     [7.70072]      [-0.033363]
  11 │    0.01      0.1     0.01     11  [0.48177]   -12.5107   [5.36565]      [8.86902]     [7.91162]      [-0.0315314]
  12 │    0.01      0.1     0.01     12  [0.450238]  -12.7398   [5.00355]      [8.48248]     [8.08286]      [-0.029836]
  13 │    0.01      0.1     0.01     13  [0.420402]  -12.8374   [3.90748]      [8.02498]     [8.15471]      [-0.0281021]
  14 │    0.01      0.1     0.01     14  [0.3923]    -12.9241   [2.93999]      [7.51648]     [8.1596]       [-0.0263136]
  15 │    0.01      0.1     0.01     15  [0.365987]  -13.0258   [1.86548]      [6.95138]     [8.1596]       [-0.0243353]
  16 │    0.01      0.1     0.01     16  [0.341651]  -13.1445   [0.578173]     [6.31406]     [8.1596]       [-0.0221042]
  17 │    0.01      0.1     0.01     17  [0.319547]  -13.0476   [-0.471392]    [5.63551]     [8.1596]       [-0.0197287]
  18 │    0.01      0.1     0.01     18  [0.299819]  -13.005    [-2.23219]     [4.84874]     [8.1596]       [-0.0169744]
  19 │    0.01      0.1     0.01     19  [0.282844]  -12.9864   [-2.92041]     [4.07183]     [8.16329]      [-0.0142514]
  20 │    0.01      0.1     0.01     20  [0.268593]  -12.9353   [-4.70996]     [3.19365]     [8.3035]       [-0.011083]
  21 │    0.01      0.1     0.01     21  [0.25751]   -12.8462   [-5.47252]     [2.32703]     [8.51995]      [-0.0079723]
  22 │    0.01      0.1     0.01     22  [0.249537]  -12.6467   [-6.60268]     [1.43406]     [8.8707]       [-0.00481491]
  23 │    0.01      0.1     0.01     23  [0.244723]  -12.7247   [-6.33227]     [0.657428]    [9.18297]      [-0.00216948]
  24 │    0.01      0.1     0.01     24  [0.242553]  -12.8298   [-6.70496]     [-0.0788115]  [9.54071]      [0.000255152]
  25 │    0.01      0.1     0.01     25  [0.242808]  -12.6414   [-6.96349]     [-0.767279]   [9.9302]       [0.00243486]
  26 │    0.01      0.1     0.01     26  [0.245243]  -12.6024   [-7.14236]     [-1.40479]    [10.341]       [0.00436846]
  27 │    0.01      0.1     0.01     27  [0.249612]  -12.6806   [-6.39765]     [-1.90407]    [10.6469]      [0.00583541]
  28 │    0.01      0.1     0.01     28  [0.255447]  -12.8766   [-5.42621]     [-2.25629]    [10.8349]      [0.0068546]
  29 │    0.01      0.1     0.01     29  [0.262302]  -12.8672   [-4.98699]     [-2.52936]    [10.9752]      [0.0076349]
  30 │    0.01      0.1     0.01     30  [0.269936]  -12.69     [-5.36535]     [-2.81296]    [11.1534]      [0.00842287]
  31 │    0.01      0.1     0.01     31  [0.278359]  -12.8472   [-3.85241]     [-2.9169]     [11.1902]      [0.00871971]
  32 │    0.01      0.1     0.01     32  [0.287079]  -13.003    [-2.67633]     [-2.89284]    [11.1902]      [0.0086478]
  33 │    0.01      0.1     0.01     33  [0.295727]  -12.897    [-2.41077]     [-2.84464]    [11.1902]      [0.00850369]
  34 │    0.01      0.1     0.01     34  [0.304231]  -12.9895   [-1.61242]     [-2.72141]    [11.1902]      [0.00813533]
  35 │    0.01      0.1     0.01     35  [0.312366]  -13.047    [-0.86709]     [-2.53598]    [11.1902]      [0.007581]
  36 │    0.01      0.1     0.01     36  [0.319947]  -13.0287   [-0.666515]    [-2.34904]    [11.1902]      [0.00702215]
  37 │    0.01      0.1     0.01     37  [0.326969]  -13.0253   [-0.155275]    [-2.12966]    [11.1902]      [0.00636635]
  38 │    0.01      0.1     0.01     38  [0.333335]  -13.0209   [0.128243]     [-1.90387]    [11.1902]      [0.00569138]
  39 │    0.01      0.1     0.01     39  [0.339027]  -12.9271   [0.0294629]    [-1.71054]    [11.1902]      [0.00511343]
  40 │    0.01      0.1     0.01     40  [0.34414]   -12.9156   [0.367215]     [-1.50276]    [11.1902]      [0.00449232]
  41 │    0.01      0.1     0.01     41  [0.348633]  -12.986    [0.659557]     [-1.28653]    [11.1902]      [0.00384592]
  42 │    0.01      0.1     0.01     42  [0.352478]  -13.0485   [1.09632]      [-1.04824]    [11.1902]      [0.0031336]
  43 │    0.01      0.1     0.01     43  [0.355612]  -12.9853   [1.28838]      [-0.814582]   [11.1902]      [0.00243509]
  44 │    0.01      0.1     0.01     44  [0.358047]  -12.9752   [1.5468]       [-0.578444]   [11.1902]      [0.00172918]
  45 │    0.01      0.1     0.01     45  [0.359776]  -13.0076   [1.541]        [-0.366499]   [11.1902]      [0.0010956]
  46 │    0.01      0.1     0.01     46  [0.360872]  -12.9594   [1.35066]      [-0.194784]   [11.1902]      [0.000582281]
  47 │    0.01      0.1     0.01     47  [0.361454]  -12.956    [1.4433]       [-0.0309752]  [11.1902]      [9.25964e-5]
  48 │    0.01      0.1     0.01     48  [0.361547]  -12.956    [1.60799]      [0.132921]    [11.1902]      [-0.00039735]
  49 │    0.01      0.1     0.01     49  [0.361149]  -13.0046   [1.42665]      [0.262294]    [11.1902]      [-0.000784094]
  50 │    0.01      0.1     0.01     50  [0.360365]  -13.0956   [1.82696]      [0.41876]     [11.1902]      [-0.00125183]
  51 │    0.01      0.1     0.01     51  [0.359114]  -12.9629   [1.36866]      [0.51375]     [11.1902]      [-0.00153579]
  52 │    0.01      0.1     0.01     52  [0.357578]  -12.9446   [1.60631]      [0.623007]    [11.1902]      [-0.0018624]
  53 │    0.01      0.1     0.01     53  [0.355715]  -13.0111   [0.884743]     [0.64918]     [11.1902]      [-0.00194064]
  54 │    0.01      0.1     0.01     54  [0.353775]  -12.9621   [0.692179]     [0.65348]     [11.1902]      [-0.0019535]
  55 │    0.01      0.1     0.01     55  [0.351821]  -13.0818   [1.19488]      [0.70762]     [11.1902]      [-0.00211534]
  56 │    0.01      0.1     0.01     56  [0.349706]  -13.0126   [0.808823]     [0.717741]    [11.1902]      [-0.0021456]
  57 │    0.01      0.1     0.01     57  [0.34756]   -13.0746   [0.809506]     [0.726917]    [11.1902]      [-0.00217303]
  58 │    0.01      0.1     0.01     58  [0.345387]  -13.0823   [0.780243]     [0.73225]     [11.1902]      [-0.00218897]
  59 │    0.01      0.1     0.01     59  [0.343198]  -13.0186   [0.499421]     [0.708967]    [11.1902]      [-0.00211937]
  60 │    0.01      0.1     0.01     60  [0.341079]  -13.0456   [0.526001]     [0.69067]     [11.1902]      [-0.00206467]
  61 │    0.01      0.1     0.01     61  [0.339014]  -13.0493   [0.341787]     [0.655782]    [11.1902]      [-0.00196038]
  62 │    0.01      0.1     0.01     62  [0.337054]  -12.8872   [-0.478001]    [0.542404]    [11.1902]      [-0.00162145]
  63 │    0.01      0.1     0.01     63  [0.335432]  -12.9422   [0.243911]     [0.512554]    [11.1902]      [-0.00153222]
  64 │    0.01      0.1     0.01     64  [0.3339]    -12.9122   [-0.1211]      [0.449189]    [11.1902]      [-0.00134279]
  65 │    0.01      0.1     0.01     65  [0.332557]  -13.0425   [0.377574]     [0.442027]    [11.1902]      [-0.00132139]
  66 │    0.01      0.1     0.01     66  [0.331236]  -12.9999   [-0.0737422]   [0.39045]     [11.1902]      [-0.0011672]
  67 │    0.01      0.1     0.01     67  [0.330069]  -13.0888   [0.252431]     [0.376648]    [11.1902]      [-0.00112594]
  68 │    0.01      0.1     0.01     68  [0.328943]  -12.9758   [-0.122331]    [0.32675]     [11.1902]      [-0.00097678]
  69 │    0.01      0.1     0.01     69  [0.327966]  -13.0302   [-0.257161]    [0.268359]    [11.1902]      [-0.000802227]
  70 │    0.01      0.1     0.01     70  [0.327164]  -12.8849   [-0.604668]    [0.181057]    [11.1902]      [-0.000541246]
  71 │    0.01      0.1     0.01     71  [0.326623]  -12.948    [-0.474788]    [0.115472]    [11.1902]      [-0.00034519]
  72 │    0.01      0.1     0.01     72  [0.326277]  -12.9255   [-0.293566]    [0.0745683]   [11.1902]      [-0.000222913]
  73 │    0.01      0.1     0.01     73  [0.326054]  -12.9125   [-0.179361]    [0.0491754]   [11.1902]      [-0.000147004]
  74 │    0.01      0.1     0.01     74  [0.325907]  -13.0816   [-0.162379]    [0.02802]     [11.1902]      [-8.37624e-5]
  75 │    0.01      0.1     0.01     75  [0.325824]  -12.9071   [-0.817104]    [-0.0564923]  [11.1902]      [0.000168877]
  76 │    0.01      0.1     0.01     76  [0.325993]  -13.0681   [-0.239916]    [-0.0748347]  [11.1902]      [0.000223709]
  77 │    0.01      0.1     0.01     77  [0.326216]  -13.0101   [-0.502281]    [-0.117579]   [11.1902]      [0.000351489]
  78 │    0.01      0.1     0.01     78  [0.326568]  -13.1136   [0.0883187]    [-0.0969895]  [11.1902]      [0.000289938]
  79 │    0.01      0.1     0.01     79  [0.326858]  -13.0495   [0.0585228]    [-0.0814383]  [11.1902]      [0.00024345]
  80 │    0.01      0.1     0.01     80  [0.327101]  -13.0158   [-0.62517]     [-0.135811]   [11.1902]      [0.000405991]
  81 │    0.01      0.1     0.01     81  [0.327507]  -12.981    [0.289529]     [-0.0932774]  [11.1902]      [0.000278841]
  82 │    0.01      0.1     0.01     82  [0.327786]  -13.0029   [-0.115089]    [-0.0954586]  [11.1902]      [0.000285361]
  83 │    0.01      0.1     0.01     83  [0.328071]  -12.9877   [-0.181564]    [-0.104069]   [11.1902]      [0.000311102]
  84 │    0.01      0.1     0.01     84  [0.328382]  -12.9942   [-0.170822]    [-0.110744]   [11.1902]      [0.000331056]
  85 │    0.01      0.1     0.01     85  [0.328714]  -13.0454   [-0.264889]    [-0.126159]   [11.1902]      [0.000377136]
  86 │    0.01      0.1     0.01     86  [0.329091]  -13.0744   [0.0157432]    [-0.111969]   [11.1902]      [0.000334716]
  87 │    0.01      0.1     0.01     87  [0.329425]  -13.0618   [-0.218752]    [-0.122647]   [11.1902]      [0.000366638]
  88 │    0.01      0.1     0.01     88  [0.329792]  -12.9788   [-0.40391]     [-0.150773]   [11.1902]      [0.000450718]
  89 │    0.01      0.1     0.01     89  [0.330243]  -13.0376   [-0.0941277]   [-0.145109]   [11.1902]      [0.000433784]
  90 │    0.01      0.1     0.01     90  [0.330677]  -13.0262   [0.101687]     [-0.120429]   [11.1902]      [0.000360008]
  91 │    0.01      0.1     0.01     91  [0.331037]  -12.914    [-0.424442]    [-0.15083]    [11.1902]      [0.000450889]
  92 │    0.01      0.1     0.01     92  [0.331487]  -13.0794   [-0.00773753]  [-0.136521]   [11.1902]      [0.000408113]
  93 │    0.01      0.1     0.01     93  [0.331896]  -13.0045   [-0.222839]    [-0.145153]   [11.1902]      [0.000433916]
  94 │    0.01      0.1     0.01     94  [0.332329]  -13.0518   [-0.119833]    [-0.142621]   [11.1902]      [0.000426347]
  95 │    0.01      0.1     0.01     95  [0.332756]  -12.9658   [0.121323]     [-0.116227]   [11.1902]      [0.000347445]
  96 │    0.01      0.1     0.01     96  [0.333103]  -13.0481   [0.393995]     [-0.0652044]  [11.1902]      [0.00019492]
  97 │    0.01      0.1     0.01     97  [0.333298]  -12.9585   [-0.0774373]   [-0.0664277]  [11.1902]      [0.000198577]
  98 │    0.01      0.1     0.01     98  [0.333497]  -12.9809   [-0.215603]    [-0.0813452]  [11.1902]      [0.000243171]
  99 │    0.01      0.1     0.01     99  [0.33374]   -13.0133   [0.126927]     [-0.060518]   [11.1902]      [0.000180911]
 100 │    0.01      0.1     0.01    100  [0.333921]  -12.9609   [0.0137386]    [-0.0530924]  [11.1902]      [0.000158713]
 101 │    0.01      0.1     0.01    101  [0.33408]   -13.0542   [-0.0760604]   [-0.0553892]  [11.1902]      [0.000165579]
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
![](docs/README-41.png)

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
  last value: -13.01383286936652
  last params: [0.33330557657068277]
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
![](docs/README-45.png)

### Importance sampling

Finally, we have a good estimate for the parameter to use with importance
sampling. Gutzwiller.jl provides `AnsatzSampling`, which is similar to
`GutzwillerSampling` from Rimu, but can be used with different ansatze.

````julia
p = grad_result.param[end]
G = AnsatzSampling(H, ansatz, p)
````

````
Gutzwiller.AnsatzSampling{Float64, 1, Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0), Gutzwiller.GutzwillerAnsatz{Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, Float64, Rimu.Hamiltonians.HubbardReal1D{Float64, Rimu.BitStringAddresses.BoseFS{10, 10, Rimu.BitStringAddresses.BitString{19, 1, UInt32}}, 2.0, 1.0}}(HubbardReal1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=2.0, t=1.0)), [0.33407952896097876])
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
  mean = -10.96 ± 0.5
  with uncertainty of ± 0.06453926876234162
  from 31 blocks after 5 transformations (k = 6).

````

````julia
prob_sampled = ProjectorMonteCarloProblem(G; target_walkers=15, last_step=2000)
sim_sampled = solve(prob_sampled)
shift_estimator(sim_sampled; skip=1000)
````

````
BlockingResult{Float64}
  mean = -12.72 ± 0.36
  with uncertainty of ± 0.03299381777618156
  from 62 blocks after 4 transformations (k = 5).

````

Note that the lower energy estimate in the sampled case is probably due to a reduced
population control bias. The effect importance sampling has on the statistic can be more
dramatic for larger systems and beter choices of anstaze.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
