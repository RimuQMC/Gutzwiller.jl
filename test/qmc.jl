using Gutzwiller
using KrylovKit
using Random
using Rimu
using Test

@testset "qmc" begin
    Random.seed!(1336)

    for H in (
        HubbardReal1D(
            near_uniform(BoseFS{5,5}); t=0.1
        ),
        HubbardMom1D(
            BoseFS(10, 5 => 3); u=4
        ),
        HubbardRealSpace(
            FermiFS2C((1,0,0,1), (1,1,0,0)); geometry=PeriodicBoundaries(2,2)
        ),
        HubbardRealSpace(
            BoseFS(1,1,1,1,0,1); u=6.0
        ),
    )
        @testset "$H w/ eigenvector" begin
            # Check that the samples are taken from the correct distribution.
            res = eigsolve(H, PDVec(starting_address(H) => 1.0), 1, :SR)
            E0 = res[1][1]
            groundstate = res[2][1]
            groundstate_sq = normalize!(map(abs2, values(groundstate)))

            res1e6 = kinetic_vqmc(H, VectorAnsatz(groundstate), Float64[]; samples=1e6)
            res1e7 = kinetic_vqmc(H, VectorAnsatz(groundstate), Float64[]; samples=1e7)

            sampled1e6 = DVec(res1e6)
            sampled1e7 = DVec(res1e7)

            len = size(DataFrame(res1e6), 1)
            @test 0.9e6 < len < 1.1e6

            val1e6, grad1e6 = val_and_grad(res1e6)
            val1e7, grad1e7 = val_and_grad(res1e7)

            @test length(grad1e6) == length(grad1e7) == 0
            @test val1e6 ≈ E0
            @test val1e7 ≈ E0

            @test dot(sampled1e6, groundstate_sq) ≈ 1 atol=1e-2
            @test dot(sampled1e7, groundstate_sq) ≈ 1 rtol=1e-2

            @test local_energy_estimator(res1e6).mean ≈ E0 rtol=1e-6
            @test local_energy_estimator(res1e7).mean ≈ E0 rtol=5e-7

            # Contination run
            kinetic_vqmc!(res1e6)
            @test size(DataFrame(res1e6), 1) == 2 * len
        end

        @testset "$H w/ Gutzwiller" begin
            ansatz = GutzwillerAnsatz(H)
            vector = PDVec(ansatz, 0.5)
            vector_sq = normalize!(map(abs2, values(vector)))

            E_v = rayleigh_quotient(H, vector)

            res = kinetic_vqmc(H, ansatz, 0.5; samples=1e6)

            # Wrapper
            val1, grad1 = val_and_grad(res)
            val2, grad2 = val_and_grad(KineticVQMC(H, ansatz; samples=1e6), 0.5)
            @test val1 ≈ val2 rtol=1e-1
            @test grad1 ≈ grad2 rtol=1e-1

            # Estimator w/ error
            est = local_energy_estimator(res)
            @test est.mean - 3est.err < E_v < est.mean + 3est.err

            # Sampling vector squared
            sampled = normalize!(DVec(res))
            @test dot(sampled, vector_sq) ≈ 1 rtol=1e-4
        end
    end
end
