using Test

include("../src/ClusterSimulation.jl")


@testset "Heterogeneous Effect Test" begin
    beta1 = rand(Normal(2,4),50)
    hep = hePopulation(50,beta1,randn(50))

    @testset "Simulation test" begin
        @test isa(hep, hePopulation)

        hes = sample(hep,100)
        @test isa(hes,clSample)
    end
    @testset "Consistency Test" begin
        betahat = (bols.(sample(hep,1000).allclusters))
        consistentat01 = .≈(permutedims(stack(betahat))[:,2] - beta1 , 0; atol=0.1)
        @test all(consistentat01)
    end
end

@testset "cl Test" begin
    @testset "Simulation test" begin
        beta2 = 2

        clp = clPopulation(50, randn(50), beta2)
        @test isa(slp,clPopulation)

        cls = sample(s,1000)
        @test isa(cls, clSample)
    end
    @testset "Consistency Test" begin
        consistentat04 = .≈(permutedims(stack(bols.(cls.allclusters)))[:,2] .- beta2, 0 ; atol=0.4)
        @test all(consistentat04)
    end

    @testset "Montecarlo" begin
        montecarlo(hep,100,100)
        #FINISCI DI SCRIVERE IL TEST PER VEDERE SE LA MONTECARLO FUZNIONA
    end

end


beta1 = rand(Normal(2,4),50)
hep = hePopulation(50,beta1,randn(50))
hes = sample(hep,100)

ccc = hes.allclusters[1]