using Test
using Gutzwiller
using Optim
using NLSolversBase
using Rimu

@testset "Gradient descent vs amsgrad" begin
    @testset "Gutzwiller, LocalEnergyEvaluator" begin
        H = HubbardReal1D(BoseFS(1,1,1,1))
        ansatz = GutzwillerAnsatz(H)
        le = LocalEnergyEvaluator(H, ansatz)

        opt_res = optimize(NLSolversBase.only_fg!(le), [0.5])
        gd_res = gradient_descent(le, [0.5])
        ams_res = amsgrad(le, [0.5]; maxiter=1000)

        @test gd_res.converged
        @test ams_res.converged

        @test gd_res.param[end] ≈ opt_res.minimizer atol=1e-3
        @test ams_res.param[end] ≈ opt_res.minimizer atol=1e-3
    end

    @testset "RelativeJastrow, KineticVQMC" begin
        H = HubbardReal1D(BoseFS(1,1,1,1))
        ansatz = RelativeJastrowAnsatz(H)
        le = LocalEnergyEvaluator(H, ansatz)
        qmc = KineticVQMC(H, ansatz)

        opt_res = optimize(NLSolversBase.only_fg!(le), [0.5, 0.5])
        gd_res = gradient_descent(qmc, [0.5, 0.5]; maxiter=100)
        ams_res = amsgrad(qmc, [0.5, 0.5]; maxiter=100)

        @test gd_res.value[end] ≈ opt_res.minimum atol=1e-3
        @test ams_res.value[end] ≈ opt_res.minimum atol=1e-3
    end
end

@testset "Fixed params" begin
    H = HubbardReal1D(BoseFS(1,1,1,1))
    ansatz = DensityProfileAnsatz(H)
    le = LocalEnergyEvaluator(H, ansatz)

    # Since the density in HubbardReal1D is constant, they should all converge to the fixed
    # value.
    trace = amsgrad(
        le, [rand(), rand(), 0.75, rand()];
        fix_params=(false,false,true,false), maxiter=1000
    )
    @test trace.param[end] ≈ [0.75, 0.75, 0.75, 0.75] atol=1e-1
end

@testset "Continuations" begin
    H = HubbardReal1D(BoseFS(1,1,1,1))
    ansatz = RelativeJastrowAnsatz(H)
    le = LocalEnergyEvaluator(H, ansatz)

    trace1 = amsgrad(le, [0.5, 0.5]; maxiter=100)

    trace2a = amsgrad(le, [0.5, 0.5]; maxiter=50)
    trace2b = amsgrad(
        le, trace2a.param[end];
        first_moment_init=trace2a.first_moment[end],
        second_moment_init=trace2a.second_moment[end],
        maxiter=50,
    )
    trace2c = amsgrad(
        le, trace2a.param[end];
        maxiter=50,
    )
    trace2d = amsgrad(le, trace2a; maxiter=50)

    # trace2b continues stably with small deviations, while trace2c wobbles violently at the
    # beginning
    @test trace1.param ≈ [trace2a.param[1:end-1]; trace2b.param] atol=1e-2
    @test trace1.param ≉ [trace2a.param[1:end-1]; trace2c.param] atol=1e-2
    @test trace2b.param == trace2d.param
end
