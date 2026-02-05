module Gutzwiller

using DataFrames
using Folds
using Measurements
using ProgressMeter
using Parameters
using Rimu
using Statistics
using StatsBase
using StaticArrays
using SpecialFunctions
using Tables
using VectorInterface

export num_parameters, val_and_grad, val_err_and_grad
include("ansatz/abstract.jl")
export VectorAnsatz
include("ansatz/vector.jl")
export GutzwillerAnsatz
include("ansatz/gutzwiller.jl")
export ExtendedGutzwillerAnsatz
include("ansatz/extgutzwiller.jl")
export MultinomialAnsatz
include("ansatz/multinomial.jl")
export JastrowAnsatz, RelativeJastrowAnsatz
include("ansatz/jastrow.jl")
export DensityProfileAnsatz
include("ansatz/densityprofile.jl")
export CombinationAnsatz
include("ansatz/combination.jl")

export LocalEnergyEvaluator
include("localenergy.jl")
export AnsatzSampling
include("AnsatzSampling.jl")

export kinetic_vqmc, kinetic_vqmc!, local_energy_estimator, KineticVQMC
include("kinetic-vqmc/state.jl")
include("kinetic-vqmc/qmc.jl")
include("kinetic-vqmc/wrapper.jl")

export gradient_descent, amsgrad
include("amsgrad.jl")


end
