using ForwardDiff
using DataFrames
using Gutzwiller
using KrylovKit
using Rimu
using Random
using StatsBase
using Test

@testset "time series resampling" begin
    times = rand(100)
    vals = rand(100)

    μ = mean(vals, FrequencyWeights(times))

    for k in (1, 2, 4, 8, 10, 16, 32, 64, 100, 200, 1000)
        @test mean(Gutzwiller.resample(times, vals; len=k)) ≈ μ
    end
end

include("ansatz.jl")
include("qmc.jl")
include("sampling.jl")
include("amsgrad.jl")
include("doctests.jl")
