using Test

include("../src/ClusterSimulation.jl")


@testset "Heterogeneous Effect Test" begin
    beta1 = rand(Normal(2,4),50)
    pp = hePopulation(50,beta1,randn(50))
    @test isa(pp, hePopulation)

    ss = sample(pp,100)
    @test isa(ss,clSample)

    betahat = (bols.(sample(pp,1000).allclusters))
    consistentat01 = .≈(permutedims(stack(betahat))[:,2] - beta1 , 0; atol=0.1)
    @test all(consistentat01)
end