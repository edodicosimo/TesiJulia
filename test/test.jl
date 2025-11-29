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

    @testset "Expetcet score is 0" begin
        X = getRegressorNoIntercept.(hes.allclusters)
        U =  getU.(hes.allclusters)
        iszero = .≈(mean.(broadcast((x,u) -> x .* u, X,U)),0;atol=0.2)
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
        @test isa(clp,clPopulation)

        cls = sample(clp,1000)
        @test isa(cls, clSample)
    
        @testset "Consistency Test" begin
            consistentat04 = .≈(permutedims(stack(bols.(cls.allclusters)))[:,2] .- beta2, 0 ; atol=0.4)
            @test all(consistentat04)
        end
    end

    @testset "Montecarlo" begin
        montecarlo(hep,100,100)
        #TODO FINISCI DI SCRIVERE IL TEST PER VEDERE SE LA MONTECARLO FUZNIONA
    end

end

beta1 = rand(Normal(2,4),500)
hep = hePopulation(50,beta1,randn(500))
hes = sample(hep,1000)

whiteAvar(hes)

getallX(hes)

ccc = hes.allclusters[1]
computeResidual(ccc)
WhiteAvar(ccc)
computeExpectedSig.(hes.allclusters)
stack(WhiteAvar.(hes.allclusters))
innerBCrve(ccc)
CRVE(hes)
(montecarlo(hep,1000,100))

permutedims(stack(bols.(hes.allclusters)))[:,2]

df = consistencyNg(hep)


using Plots

plot(
    df.Ng,
    df.Mean,
    ribbon = df.SD,
    xlabel = "Ng",
    ylabel = "BetaHat",
    legend = false,
    lw = 2,
    fillalpha = 0.3,
)
 ## Test that the score is 0 in every cluster

x = ccc.X[:,2]
u = ccc.u 

print( ccc.df )

mean(x)
std(x)

histogram(
    x,
    bins = 5000
)

histogram2d(x, u,
    nbins=50,
    xlabel="X",
    ylabel="u",
    title="2D Histogram")

mean( x .* u )

reduce(vcat, getRegressorNoIntercept.( hes.allclusters ) )

function getRegressorNoIntercept(c::clCluster)
    c.X[:,2]
end

function checkScore(s::clSample)
    getRegressorNoIntercept.(s.allclusters) .* getU.(s.allclusters)
end

function getU(c::clCluster)
    c.u
end

X = getRegressorNoIntercept.(hes.allclusters)
U =  getU.(hes.allclusters)

histogram(
    mean.(broadcast((x,u) -> x .* u, X,U)),
    bins=50
    )