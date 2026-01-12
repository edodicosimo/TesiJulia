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
m = montecarlo(pop,1000,Ng,false,true) 

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


white = getindex.(WhiteAvar.(sam.allclusters),2,2)
histogram(white .- var(M, dims = 2))
print("The average across clusters difference between the estimated variance using white and the Montecarlo variance is: ", mean(white - var(M,dims = 2)))


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

v = vcat(collect(1:5:50), collect(50:10:250),collect(250:100:1000))
l = length(v)

betaDict = Dict()
for G in v
    varianceBeta = []
    for _ in 1:1000
        beta = rand(β,G)
        barbeta = mean(beta)
        push!(varianceBeta, barbeta)
    end
    beta = rand(β,G)
    estimatedVar = (1 / (G - 1)) * sum((beta .- mean(beta)).^2)/G 
    var(varianceBeta)
    betaDict[G] = estimatedVar - var(varianceBeta)
end 


Gs = sort(collect(keys(betaDict)))
vals = [betaDict[G] for G in Gs]

plot(
    Gs,
    vals,
    seriestype = :line,
    marker = :circle,
    linewidth = 2,
    markersize = 1,
    alpha = 0.9,
    xlabel = L"G",
    ylabel = L"\widehat{\mathrm{Var}}(\bar\beta) - \mathrm{Var}(\bar\beta)",
    title = "Variance Estimation Error vs Number of Clusters",
    legend = false,
    grid = :on
)

hline!([0.0], linestyle = :dash, linewidth = 1.5) 

histogram(
    varianceOfBeta,
    bins = 100
)


hatBetaG = getBeta(bols.(sam.allclusters))
sqrt(G) * mean(hatBetaG .- 2)


#### Scomposizione tra eg e dg
### Questo fa vedere come e_g tende a zero all'aumentare di Ng
Ngs = [10,100,200,300,1000]
dNg = Dict()
for Ng in Ngs
    d = Dict(i => [] for i in 1:G)
    for _ in 1:1000
        sample1 = sample(pop,Ng)
        eg = getBeta(bols.(sample1.allclusters)) .- beta1
        for g in 1:G
            push!(d[g],eg[g])
        end
    end
    vareg = mean([var(e) for e in values(d)])
    dNg[Ng] = vareg
end
xs = sort(collect(keys(dNg)))
ys = [dNg[n] for n in xs]

plot(
    xs,
    ys;
    xlabel = L"N_g",
    ylabel = L"\mathbb{E}[\mathrm{Var}(e_g)]",
    title = "Decay of Within-Cluster Sampling Error",
    lw = 3,
    ms = 5,
    marker = :circle,
    legend = false,
    grid = :on,
    dpi = 300,
)

# Questo fa vedere la varianza di dg
mus = []
for _ in 1:1000
    beta = rand(β,G)
    dg = beta .- 2
    m = mean(dg)
    push!(mus,m)
end
sqrt(var(mus) + 0.0103885)

Gs = [5, 10, 20, 50, 100]
dG = Dict()

for G in Gs
    mus = Float64[]
    for _ in 1:1000
        beta = rand(β, G)
        dg = beta .- 2
        m = mean(dg)
        push!(mus, m)
    end
    dG[G] = var(mus)
end

xs = sort(collect(keys(dG)))
ys = [dG[g] for g in xs]

plot(
    xs,
    ys;
    xlabel = L"G",
    ylabel = L"\mathrm{Var}(\overline{d}_g)",
    title = "Variance of Mean Deviations as G Increases",
    lw = 3,
    ms = 5,
    marker = :circle,
    legend = false,
    grid = :on,
    dpi = 300,
)

sqrt(whiteAvar(sam))
sqrt(CRVE(sam))


#### La varianza di betapols non è la media delle varianze di beta g se ho resampling
         
allBetaPols = []
allBetaG = []

for _ in 1:1000
    beta = rand(β, G)
    population1 = hePopulation(G, beta, randn(G))
    sample1 = sample(population1, Ng)
    bpols = bols(sample1)[2]
    hatBetag = getBeta(bols.(sample1.allclusters)) .- beta
    push!(allBetaPols,bpols)
    push!(allBetaG,hatBetag)
end

mean(var(reduce(hcat,allBetaG), dims=2))

var(allBetaPols)


###########
ng = 30
B1 = 1
B2 = 5
B = mean([B1,B2])

X1 = randn(ng)
X2 = randn(ng)

X = reduce(vcat, [X1,X2])

U1 = randn(ng)
U2 = randn(ng)

S1 = X1 .* U1
S2 = X2 .* U2

Y1 = X1 .* B1 .+ U1  
Y2 = X2 .* B2 .+ U2

FakeU1 = Y1 .- X1 .* B
FakeU2 = Y2 .- X2 .* B 

sigmaSrB = var([mean(FakeU1), mean(FakeU2)]) 
sigmaSrW = mean([var(FakeU1),var(FakeU2)])
sigmaSrT = var(vcat(FakeU1,FakeU2))


mean(FakeU1)
mean(FakeU2)
x = -10:0.1:10

Y = reduce(vcat,[Y1,Y2])

p = scatter(Y1,X1)
scatter!(Y2,X2)
# plot!(p, x, B1 .* x, label="B1")
# plot!(p, x, B2 .* x, label="B2")

