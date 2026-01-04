#LOAD LIBRARIES
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Plots
using LaTeXStrings

abstract type Population end


struct clPopulation <: Population
    G::Integer
    mu::Vector{Float64}
    β::Float64
end

struct hePopulation <: Population
    G::Int64
    βg::Vector{Float64}
    cg::Vector{Float64}
end

struct clCluster
    Ng :: Integer
    df :: DataFrame
    X :: Matrix
    y :: Vector
    u :: Vector
end

struct clSample
    allclusters::Array{clCluster,1}
    N::Integer
end

function Base.show(io::IO, c::clSample)
    print(io, "Cluster sample struct")
end

function Base.show(io::IO, p::Population)
    s = ""
    if isa(p, hePopulation)
        s = "with"
    else
        s = "without"
    end 
    G = p.G
    print(io, "Population with $G clusters and $s heterogeneous effect")
end

function sample(p::clPopulation, Ng::Int)
    N = p.G * Ng
    allc = simulateclCluster.(p.mu,Ng, p.β)
    sample = clSample(allc, N)
end

function sample(p::hePopulation, Ng::Int)
    N = p.G * Ng
    allc = simulateHeCluster.(p.βg,p.cg,Ng)
    sample = clSample(allc,N)
end

function simulateclCluster(mu_g::Float64, Ng::Int64, β::Float64)
    s = sqrt.(abs.(mu_g).+10) #questo 10 qua è v_0
    a = mu_g ./ s
    b = sqrt.(max.(s .^ 2 .- a.^2, 0))
    Z1 = randn(Ng)
    Z2 = randn(Ng)
    Xig = [ones(Ng) s .* Z1]
    uig = a .* Z1 .+ b .* Z2
    beta = [1, β]
    y_ig =  Xig * beta .+ uig
    df = DataFrame(
        "Xig" => Xig[:,2],
        "y_ig" => y_ig,
        "uig" => uig
        )
    c = clCluster(Ng, df, Xig, y_ig, uig)
end

"""
Simulate a cluster given a scalar beta and cg.
"""
function simulateHeCluster(beta::Float64, cg::Float64, Ng::Int64)
    Xig = [ones(Ng) randn(Ng)]
    betag = [1, beta]
    vig = randn(Ng)
    uig = vig .+ cg
    y_ig = Xig * betag .+ uig
    df = DataFrame(
        "Xig" => Xig[:,2],
        "y_ig" => y_ig,
        "uig" => uig
    )
    c = clCluster(Ng, df, Xig, y_ig, uig)
end



function bols(c::clCluster)
    (inv(c.X' * c.X) * (c.X' * c.y))
end

function bols(s::clSample)
    X = getallX(s)
    Y = reduce(vcat, getallY(s))
    betahat = inv(X' * X) * (X'Y)
end

function getX(c::clCluster)
    c.X
end


function getY(c::clCluster)
    c.y
end


function getallX(s::clSample)
    reduce(vcat,getX.(s.allclusters)) 
end


function getallY(s::clSample)
    getY.(s.allclusters) 
end

function getRegressorNoIntercept(c::clCluster)
    c.X[:,2]
end

function checkScore(s::clSample)
    X = getRegressorNoIntercept.(s.allclusters)
    U = getU.(s.allclusters)
    Ng = length(s.allclusters[1].X)
    broadcast((x,u)->x'*u / Ng,X,U)
end

function getU(c::clCluster)
    c.u
end


### SI POTREBBE FARE UN STRUCT MONTECARLO, MA PER ORA è IMPLEMENTATO COME UN SEMPLICE VETTORE QUINDI CI STA
"""
Run a montecarlo simulation given a population  
ARGUMENTS:  
    p : Population  
    iterations  
    Ng : The number of indiviuals to sample in each cluster  
    allbeta : boolean, if false it returns only the mean and stdev of the montecarlo, if true returns all estimated betas  
    two_stage_sampling : bool  
"""
function montecarlo(p::Population, iterations::Int64, Ng::Int64, allbeta::Bool=false, two_stage_sampling::Bool= false)
    v = clSample[]
    if two_stage_sampling
        v = populationSimulationForMontecarlo(4.0,iterations,p,Ng,2.0)
    else
        v = sample.(fill(p,iterations),Ng)
    end
    b = bols.(v)
    if allbeta
        return getBeta(b)
    else
        return (mean(getBeta(b)), std(getBeta(b)))
    end
end

function populationSimulationForMontecarlo(sigma_mu::Float64, iterations::Int64, p::clPopulation,Ng::Int64, beta::Float64)
    G = p.G
    v = rand.(fill((Normal(0,sigma_mu)),iterations),G)
    sample.(clPopulation.(G,v,beta),Ng)
end

function populationSimulationForMontecarlo(sigma_mu::Float64, iterations::Int64, p::hePopulation,Ng::Int64, beta::Float64)
    # HePopulation tiene un vettore di beta come campo
    # devo creare un vettore di popolazioni ognuna con un diverso betaG, da lì samplo, quindi non può prendere la popolazione come argomento
    # step1 creo un vettore di vettori (i betag)
    # step 2 creo una popolazione con ognugno di questi vettori
    #samplo
    β, sigma2 = mean(p.βg), std(p.βg)
    G = p.G
    cg = [randn(G) for _ in 1: iterations]
    betaDistribution = Normal(2, sigma2)
    betaVectors = [rand(betaDistribution, G) for _ in 1:iterations]
    populations = hePopulation.(G,betaVectors,cg)
    sample.(populations,Ng)
end

"""
Run a montecarlo simulation computing beta_hat in each cluster  
## Parameters:   
- p : Population  
- iterations : Int  
- Ng : Int  
## Returns:
A matrix of {iterations} rows and G columns, each one is the estimated  
parameters in cluster g
"""
function montecarloClusterWise(p::Population, iterations::Int64, Ng::Int64)
    v = fill(p,iterations)
    s = getAllClusters.(sample.(v,Ng))
    b = broadcast((a)->bols.(a),s)
    stack(getBeta.(b))
end

function getAllClusters(s::clSample)
    s.allclusters
end

function consistencyNg(p::hePopulation)
    v = vcat(collect(1:5:50), collect(50:10:250),collect(250:100:1000))
    l = length(v)
    pop = fill(p,l)
    betahat = montecarlo.(pop,100,v)
    df = DataFrame(
        "Ng" => v,
        "BetaHat" => betahat
    )
    df.Mean = first.(df.BetaHat)
    df.SD   = last.(df.BetaHat)
    return df
end

# Per fare la consistenza in G dobbiamo

#TODO TESTARE LA CONSISTENZA (IN G VS IN NG) 
#TODO TESTARE SE IL MIO MODO DI FARE LA MONTECARLO E PIU VELOCE CHE FARLO CON UN CICLO FOR

function computeExpectedSig(c::clCluster) 
    mean(c.u .* c.X)
end

function computeResidual(c::clCluster)
    betahat = bols(c)
    c.y .- c.X * betahat 
end

function computeResidual(s::clSample)
    betahat = bols(s)
    reduce(vcat,getallY(s)) .- getallX(s) * betahat
end


function WhiteAvar(c::clCluster)
    uhat = computeResidual(c)
    X = c.X
    inv(X' *  X) * X' * Diagonal(uhat .^ 2) * X * inv(X' *  X)
end

function whiteAvar(s::clSample)
    uhat = computeResidual(s)
    X = getallX(s)
    inv(X' *  X) * X' * Diagonal(uhat .^ 2) * X * inv(X' *  X)
end

function innerBCrve(c::clCluster, betahat::Vector{Float64})
    X = c.X
    uhat = c.y .- c.X * betahat  
    X' * uhat * uhat' * X
end

function CRVE(s::clSample)
    betahat = bols(s)
    B = sum(broadcast((c) -> innerBCrve(c,betahat), s.allclusters))
    X = getallX(s)
    inv(X'*X) * B * inv(X'*X)
end

function getBeta(a::Vector{Vector{Float64}})
    permutedims(stack(a))[:,2]
end 