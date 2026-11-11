using Gutzwiller
using Rimu
using Test
using KrylovKit

@testset "ansatz sampling" begin
    @testset "Rimu comparison" begin
        for (u, g) in ((6.0, 0.5), (1.0, 0.2))
            H = HubbardReal1D(BoseFS((1,1,1,0)); u)
            G1 = GutzwillerSampling(H, g)
            G2 = AnsatzSampling(H, GutzwillerAnsatz(H), g)

            @test sparse(G1) ≈ sparse(G2)
            @test LOStructure(G2) ≡ AdjointKnown()
            @test sparse(G2') ≈ sparse(G2)'
        end
    end

    @testset "ground states" begin
        H = HubbardReal1D(BoseFS((1,1,1,2,1)); u=3.0)
        E0 = eigsolve(sparse(H), 1, :SR)[1][1]

        for ansatz in (
            GutzwillerAnsatz(H),
            #ExtendedGutzwillerAnsatz(H),
            MultinomialAnsatz(H),
        )
            @testset "$ansatz" begin
                G = AnsatzSampling(H, ansatz, fill(0.5, num_parameters(ansatz)))

                @test eigsolve(sparse(G), 1, :SR)[1][1] ≈ E0
            end
        end
    end
end
