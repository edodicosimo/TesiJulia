using Test

include("../src/ClusterSimulation.jl")

# Define population and sample that will be used for testing
beta1 = rand(Normal(2,4),500)
hep = hePopulation(50,beta1,randn(500))
hes = sample(hep,1000)

@testset "Heterogeneous Effect Test" begin

    @testset "Simulation test" begin
        @test isa(hep, hePopulation)
        @test isa(hes,clSample)
    end

    @testset "Expetcet score is 0" begin
        X = getRegressorNoIntercept.(hes.allclusters)
        U =  getU.(hes.allclusters)
        iszero = .≈(mean.(broadcast((x,u) -> x .* u, X,U)),0;atol=0.2)
    end

    @testset "Consistency Test" begin
        betahat = (bols.(sample(hep,1000).allclusters))
        consistentat02 = .≈(permutedims(stack(betahat))[:,2] - beta1 , 0; atol=0.2)
        @test all(consistentat02)
    end
end

beta2 = 2
mug = rand(Normal(0,4),500)
clp = clPopulation(500, mug, beta2)
cls = sample(clp,1000)


@testset "cl Test" begin
    @testset "Simulation test" begin
        @test isa(clp,clPopulation)
        @test isa(cls, clSample)
    end
        @testset "Consistency Test" begin
            consistentat04 = .≈(permutedims(stack(bols.(cls.allclusters)))[:,2] .- beta2, 0 ; atol=0.4)
            @test all(consistentat04)
        end
    # @testset "Montecarlo" begin
    #     montecarlo(hep,100,100)
    #     #TODO FINISCI DI SCRIVERE IL TEST PER VEDERE SE LA MONTECARLO FUZNIONA
    # end

end

@testset "Variance estimators" begin
    w = whiteAvar(hes)
    @test isa(w,Matrix)
    crve = CRVE(hes)
    @test isa(crve,Matrix)
end



getallX(hes)

ccc = hes.allclusters[1]
computeResidual(ccc)
WhiteAvar(ccc)
computeExpectedSig.(hes.allclusters)
mean(WhiteAvar.(hes.allclusters))
innerBCrve(ccc)

(montecarlo(hep,1000,100))

histogram(permutedims(stack(bols.(hes.allclusters)))[:,2] - beta1)

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



X = getRegressorNoIntercept.(hes.allclusters)
U =  getU.(hes.allclusters)

histogram(
    mean.(broadcast((x,u) -> x .* u, X,U)),
    bins=50
    )


#  MONTECARLO DEI BETAG
betahatClusterWise = montecarloClusterWise(hep,1000,1000)


@time montecarloClusterWise(hep, 100, 500)

histogram(mean(betahatClusterWise, dims=2))
std(betahatClusterWise, dims=2)

diffbeta = betahatClusterWise .- beta1
histogram(mean(diffbeta,dims=2))
beta1

montecarlo(clp,100,100) 
montecarlo(hep,100,100)
