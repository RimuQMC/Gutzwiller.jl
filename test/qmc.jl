using Gutzwiller
using KrylovKit
using Random
using Rimu
using Test

# Wraps a Hamiltonian and hides its adjoint.
struct NoAdjoint{T,H} <: AbstractHamiltonian{T}
    hamiltonian::H
end
NoAdjoint(hamiltonian) = NoAdjoint{eltype(hamiltonian),typeof(hamiltonian)}(hamiltonian)
Rimu.starting_address(h::NoAdjoint) = starting_address(h.hamiltonian)
Rimu.diagonal_element(h::NoAdjoint, addr) = diagonal_element(h.hamiltonian, addr)
Rimu.num_offdiagonals(h::NoAdjoint, addr) = num_offdiagonals(h.hamiltonian, addr)
Rimu.get_offdiagonal(h::NoAdjoint, addr, i) = get_offdiagonal(h.hamiltonian, addr, i)
Rimu.LOStructure(::Type{<:NoAdjoint}) = AdjointUnknown()

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

    @testset "errors" begin
        H = HubbardReal1D(BoseFS(1, 1, 1, 1, 1); u=6.0)

        # Neighbour amplitudes overflow to Inf.
        @test_throws ErrorException kinetic_vqmc(
            H, GutzwillerAnsatz(H), -200.0; samples=100, walkers=1
        )

        # Zero amplitude at the starting address.
        vector = DVec(build_basis(H) .=> 1.0)
        vector[starting_address(H)] = 0.0
        @test_throws ErrorException kinetic_vqmc(
            H, VectorAnsatz(vector), Float64[]; samples=100, walkers=1
        )

        # All neighbours of the starting address have zero amplitude.
        vector = DVec(starting_address(H) => 1.0)
        @test_throws ErrorException kinetic_vqmc(
            H, VectorAnsatz(vector), Float64[]; samples=100, walkers=1
        )
    end

    @testset "non-Hermitian Hamiltonians" begin
        H = HubbardReal1D(BoseFS(1, 1, 1, 1, 1); u=6.0)
        ansatz = GutzwillerAnsatz(H)
        params = 0.4

        for G in (GutzwillerSampling(H, 0.3), AnsatzSampling(H, ansatz, 0.3))
            @testset "$(nameof(typeof(G)))" begin
                exact_val, exact_grad = val_and_grad(LocalEnergyEvaluator(G, ansatz), params)

                res = kinetic_vqmc(G, ansatz, params; samples=1e6)
                est = local_energy_estimator(res)
                _, grad = val_and_grad(res)

                @test est.mean - 4est.err < exact_val < est.mean + 4est.err
                @test grad ≈ exact_grad rtol=0.02
            end
        end

        @test_throws ArgumentError kinetic_vqmc(NoAdjoint(H), ansatz, params; samples=100)
    end

    @testset "equilibration" begin
        H = HubbardReal1D(BoseFS(1, 1, 1, 1, 1); u=6.0)
        ansatz = GutzwillerAnsatz(H)
        start = starting_address(H)

        # HubbardReal1D has no diagonal transitions, so the walker moves at every step.
        res = kinetic_vqmc(H, ansatz, 0.5; steps=100, walkers=2, equilibration=0)
        @test all(st -> st.addresses[1] == start, res.states)
        @test all(st -> length(st) == 100, res.states)

        res = kinetic_vqmc(H, ansatz, 0.5; steps=100, walkers=2, equilibration=1)
        @test all(st -> st.addresses[1] ≠ start, res.states)
        @test all(st -> length(st) == 100, res.states)

        # Continuation adds `steps` samples regardless of `equilibration`.
        kinetic_vqmc!(res; steps=50, equilibration=20)
        @test all(st -> length(st) == 150, res.states)

        # Default equilibration is a tenth of the steps.
        qmc = KineticVQMC(H, ansatz; samples=1000, walkers=2)
        @test qmc.equilibration == 50
        val_and_grad(qmc, 0.5)
        @test all(st -> length(st) == 500, qmc.result.states)
        qmc(0.5; equilibration=0)
        @test all(st -> length(st) == 500, qmc.result.states)
    end

    @testset "walker state" begin
        H = HubbardReal1D(BoseFS(1, 1, 1, 1, 1); u=6.0)
        res = kinetic_vqmc(H, GutzwillerAnsatz(H), 0.5; samples=1000, walkers=1)
        st = only(res.states)

        times = st.residence_times
        energies = st.local_energies
        μ = sum(times .* energies) / sum(times)
        @test mean(res) ≈ μ
        @test var(res) ≈ sum(times .* (energies .- μ) .^ 2) / sum(times)

        empty!(st)
        @test length(st.addresses) == 0
        @test length(st.residence_times) == 0
        @test length(st.local_energies) == 0
        @test length(st.grad_ratios) == 0
    end
end
