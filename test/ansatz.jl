using Test
using Gutzwiller
using Rimu
using ForwardDiff
using StaticArrays: SVector

function check_ansatz(H, ansatz, params)
    @testset "$H / $(nameof(typeof(ansatz)))" begin
        # these tests look dumb, but assuming H and params are selected properly, they do
        # make sense.
        @testset "properties" begin
            @test num_parameters(ansatz) == length(params)
            @test keytype(ansatz) == typeof(starting_address(H))
            @test valtype(ansatz) == eltype(params)
        end

        @testset "val_and_grad" begin
            addrs = [starting_address(H); rand(build_basis(H), 10)]
            for addr in addrs
                val1 = ansatz(addr, params)
                val2, grad1 = val_and_grad(ansatz, addr, params)
                grad2 = ForwardDiff.gradient(x -> ansatz(addr, x), params)

                @test val1 ≈ val2
                @test grad1 ≈ grad2 atol=1e-10 rtol=√eps(Float64)
            end
        end

        @testset "LocalEnergyEvaluator val_and_grad" begin
            le = LocalEnergyEvaluator(H, ansatz)
            val1 = le(params)
            val2, grad1 = val_and_grad(le, params)
            grad2 = ForwardDiff.gradient(le, params)

            @test val1 ≈ val2
            @test grad1 ≈ grad2 atol=1e-10 rtol=√eps(Float64)

            val_rq = rayleigh_quotient(H, PDVec(ansatz, params; basis=build_basis(H)))
            @test val_rq ≈ val1
        end
    end
end

@testset "general Ansatz tests" begin
    for H in (
        HubbardReal1D(near_uniform(BoseFS{5,5}); t=0.1),
        ExtendedHubbardReal1D(near_uniform(BoseFS{5,5}); t=0.1),
        HubbardMom1D(BoseFS(10, 5 => 3); u=4),
        HubbardRealSpace(FermiFS2C((1,0,0,1), (1,1,0,0)); geometry=PeriodicBoundaries(2,2)),
    )
        M = num_modes(starting_address(H))

        check_ansatz(H, GutzwillerAnsatz(H), rand(1))
        if H isa ExtendedHubbardReal1D
            #check_ansatz(H, ExtendedGutzwillerAnsatz(H), rand(2))
        end
        if starting_address(H) isa BoseFS
            check_ansatz(H, MultinomialAnsatz(H), rand(1))
            check_ansatz(H, GrossPitaevskiiAnsatz(H), rand(M))
            check_ansatz(H, GutzwillerAnsatz(H) + MultinomialAnsatz(H), rand(3))
        end
        if starting_address(H) isa SingleComponentFockAddress
            check_ansatz(H, JastrowAnsatz(H), rand((M * (M + 1)) ÷ 2))
            check_ansatz(H, RelativeJastrowAnsatz(H), rand(cld(M, 2)))
            check_ansatz(H, DensityProfileAnsatz(H), rand(M))
        end
    end
end

@testset "GrossPitaevskiiAnsatz" begin
    H = HubbardReal1D(BoseFS((2, 0)))
    addr = starting_address(H)
    gpa = GrossPitaevskiiAnsatz(addr)
    @test GrossPitaevskiiAnsatz(H) isa GrossPitaevskiiAnsatz{<:Any,Float64,2}
    @test GrossPitaevskiiAnsatz(addr; valtype=ComplexF64) isa GrossPitaevskiiAnsatz{<:Any,ComplexF64,2}
    @test starting_address(gpa) == addr
    @test build_basis(gpa) == build_basis(addr)
    @test repr(gpa) == "GrossPitaevskiiAnsatz($(gpa.address))"
    @test iszero(gpa(addr, SVector(0.0, 1.0)))
    @test gpa(addr, SVector(0.5, 0.5)) ≈ 0.25

    for (a, p, expected_grad) in (
        (addr, SVector(0.0, 1.0), SVector(0.0, 0.0)),
        (BoseFS((1, 1)), SVector(0.0, 1.0), SVector(sqrt(2.0), 0.0)),
        (BoseFS((1, 1)), SVector(0.0, 0.0), SVector(0.0, 0.0)),
        (addr, SVector(0.5, 0.5), SVector(1.0, 0.0)),
    )
        val, grad = val_and_grad(gpa, a, p)
        @test val ≈ gpa(a, p)
        @test grad ≈ expected_grad
    end
end

@testset "MultinomialAnsatz" begin
    @testset "is exact for u=0" begin
        for H in (
            HubbardReal1D(BoseFS((1,1,1,1,1)); u=0),
            HubbardReal1D(BoseFS((1,2,1,1)); u=0),
            HubbardRealSpace(BoseFS((1,1,1,1,0,0)); u=0, geometry=PeriodicBoundaries(2,3)),
        )
            res = eigsolve(H, DVec(starting_address(H) => 1.0), 1, :SR)
            Rimu.scale!(res[2][1], sign(first(values(res[2][1]))))
            bin = MultinomialAnsatz(H; normalize=true)
            @test LocalEnergyEvaluator(H, bin)([0.5]) ≈ res[1][1]
            @test DVec(bin, [0.5]; basis=build_basis(H)) ≈ res[2][1]
        end
    end
end
