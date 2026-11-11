using DataFrames
using Gutzwiller
using LinearAlgebra
using NLSolversBase
using Optim
using Random
using Rimu
using StaticArrays
using Test

@testset "stochastic reconfiguration" begin
    H = HubbardReal1D(BoseFS(1, 1, 1, 1, 1, 1); u=4.0)

    @testset "metric" begin
        Random.seed!(17)
        ansatz = RelativeJastrowAnsatz(H)
        params = [0.4, 0.05, 0.02]
        le = LocalEnergyEvaluator(H, ansatz)

        val, err, grad, S = val_err_grad_and_metric(le, params)
        exact_val, exact_grad = val_and_grad(le, params)
        @test val ≈ exact_val
        @test grad ≈ exact_grad
        @test err == 0
        @test S ≈ S'
        @test all(≥(-1e-12), eigvals(Symmetric(Matrix(S))))

        # The metric is the covariance of the log-derivatives over |ψ|², computed directly.
        basis = build_basis(H)
        ψ = [ansatz(addr, params) for addr in basis]
        O = [val_and_grad(ansatz, addr, params)[2] / ansatz(addr, params) for addr in basis]
        w = ψ .^ 2 / sum(ψ .^ 2)
        mean_O = sum(w .* O)
        @test S ≈ sum(w .* [(o - mean_O) * (o - mean_O)' for o in O])

        # Sampled estimates agree with the exact ones.
        qmc = KineticVQMC(H, ansatz; samples=1e6)
        s_val, s_err, s_grad, s_S = val_err_grad_and_metric(qmc, params)
        @test s_val - 4s_err < val < s_val + 4s_err
        @test s_grad ≈ grad rtol=0.05
        @test s_S ≈ S rtol=0.05
    end

    @testset "exact optimization" begin
        for (ansatz, p0) in (
            (GutzwillerAnsatz(H), [0.0]),
            (RelativeJastrowAnsatz(H), zeros(3)),
            # Translation invariance makes most of the 21 parameters redundant, so S is
            # singular.
            (JastrowAnsatz(H), zeros(21)),
        )
            @testset "$(nameof(typeof(ansatz)))" begin
                le = LocalEnergyEvaluator(H, ansatz)
                opt = optimize(
                    NLSolversBase.only_fg!(le), fill(0.3, num_parameters(ansatz)), LBFGS()
                )
                res = stochastic_reconfiguration(le, p0; maxiter=300, verbose=false)
                @test res.value[end] ≈ opt.minimum atol=1e-6
                @test le(res.param[end]) ≈ opt.minimum atol=1e-6
            end
        end
    end

    @testset "sampled optimization" begin
        Random.seed!(1234)
        # From g = 0, plain gradient descent and amsgrad overshoot into the region of large
        # g, where the gradient vanishes.
        ansatz = GutzwillerAnsatz(H)
        le = LocalEnergyEvaluator(H, ansatz)
        opt = optimize(NLSolversBase.only_fg!(le), [0.3], LBFGS())

        qmc = KineticVQMC(H, ansatz; samples=1e4)
        res = stochastic_reconfiguration(qmc, [0.0]; maxiter=100, verbose=false)
        @test le(res.param[end]) - opt.minimum < 1e-3
        @test res.iterations == 100
        @test !res.converged
    end

    @testset "fixed parameters" begin
        ansatz = RelativeJastrowAnsatz(H)
        le = LocalEnergyEvaluator(H, ansatz)
        res = stochastic_reconfiguration(
            le, [0.0, 0.05, 0.0]; fix_params=(false, true, false), maxiter=200,
            verbose=false,
        )
        @test all(p -> p[2] == 0.05, res.param)
        @test all(d -> d[2] == 0, res.param_delta)
        @test res.value[end] < res.value[1]
    end

    @testset "continuation and tables" begin
        ansatz = RelativeJastrowAnsatz(H)
        le = LocalEnergyEvaluator(H, ansatz)

        res = stochastic_reconfiguration(le, zeros(3); maxiter=50, verbose=false)
        res_a = stochastic_reconfiguration(le, zeros(3); maxiter=25, verbose=false)
        res_b = stochastic_reconfiguration(le, res_a; maxiter=25, verbose=false)
        @test [res_a.param; res_b.param] ≈ res.param
        @test final_params(res_b) ≈ final_params(res)
        @test final_params(res) ≈ res.param[end] + res.param_delta[end]

        df = DataFrame(res)
        @test size(df, 1) == res.iterations
        @test df.value == res.value
        @test all(==(0.05), df.τ)
        @test startswith(sprint(show, res), "StochasticReconfigurationResult")
    end

    @testset "deprecations" begin
        le = LocalEnergyEvaluator(H, GutzwillerAnsatz(H))
        @test_deprecated amsgrad(le, [0.5]; maxiter=1, verbose=false)
        @test_deprecated gradient_descent(le, [0.5]; maxiter=1, verbose=false)
    end
end
