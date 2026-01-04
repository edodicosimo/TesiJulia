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

Ng = 100
d = Dict()
for G in [10,50, 100, 500, 1000, 5000]
    beta = rand(Normal(2, 4), G)
    pop = hePopulation(G,beta,randn(G))
    sam = sample(pop,Ng)
        white = sqrt(whiteAvar(sam)[2,2])
        crve = sqrt(CRVE(sam)[2,2])
        stdev = montecarlo(pop,400,Ng,false,true)[2]
        dwhite = white - stdev
        dcrve = crve - stdev
        d[G] = (dwhite,dcrve)
end

iter = 500
G = 50 #
Ng = 100
β = Normal(2,4)
betaG = permutedims(reduce(hcat,[rand(β,G) for _ in 1:iter]))
μ = mean(betaG, dims=1)
Xc = betaG .- μ

Eps = (Xc' * Xc) / (iter - 1)
mean(diag(Eps)) # deve esssere uguale alla varianza di beta 

# Prendiamo un sample di beta e ci costruiamo una popolazione
beta1 = rand(β, G)
pop = hePopulation(G,beta1,randn(G))
sam = sample(pop,Ng)
M = montecarloClusterWise(pop,iter,Ng) #G rows, iter columns

plots = [
    begin
        p = histogram(
            M[i, :],
            title = "Cluster $i",
            legend = false
        )
        vline!(p, [beta1[i]], color = :red, linewidth = 2)
        p
    end
    for i in 1:size(M, 1)
]

plot(plots..., layout = (10, 5), size = (1600, 1600))

var(M, dims=2)

white = getindex.(WhiteAvar.(sam.allclusters),2,2)
mean(white - var(M,dims = 2))
mean(white)
whiteAvar(sam)[2,2]

# FACCIAMO IL PLOT DE VARI BETA_APE SE C'è RESAMPLE
beta_APE = [mean(rand(β,G)) for _ in 1:1000]
histogram(
    beta_APE,
    bins = 20,
    normalife = :pdf,
    title = L"$\beta_{\text{APE}}$ across repeated samples"
    )


varianceOfBeta = []
for _ in 1:1000
    beta = rand(β,G)
    pops = hePopulation(G,beta,randn(G))
    barBeta = mean(pops.βg)
    push!(varianceOfBeta,barBeta)
end

estimatedVar = (1 / (G - 1)) * sum((pop.βg .- mean(pop.βg)).^2)/G
estimatedStDev = sqrt(estimatedVar)
std(varianceOfBeta)

histogram(
    estimatedvariances,
    bins = 100
)


hatBetaG = getBeta(bols.(sam.allclusters))
sqrt(G) * mean(hatBetaG .- 2)

