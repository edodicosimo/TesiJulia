#LOAD LIBRARIES
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Plots


abstract type Population end
"""
    clPopulation(G, mu, β)

Container for the data-generating parameters of the clustered population.

# Parameters
- `G::Integer`: Number of clusters in the population.
- `mu::Vector{Float64}`: Cluster-level means used in the DGP.
- `β::Float64`: True regression coefficient in the population model.

Used as the base configuration for generating clustered samples in the Monte Carlo simulation.
"""
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

"""
    clCluster(Ng, df, X, y, u)

Represents a single cluster drawn from the population.

# Parameters
- `Ng::Integer`: Number of observations in the cluster.
- `df::DataFrame`: Full cluster-level dataset.
- `X::Vector`: Regressor values for the cluster.
- `y::Vector`: Outcome values for the cluster.
- `u::Vector`: Error terms generated for the cluster.

Used to store all elements associated with one cluster in the Monte Carlo simulation.
"""
struct clCluster
    Ng :: Integer
    df :: DataFrame
    X :: Matrix
    y :: Vector
    u :: Vector
end

"""
    clSample(allclusters, N)

Represents a full sample composed of multiple clusters.

# Parameters
- `allclusters::Vector{clCluster}`: Collection of all clusters included in the sample.
- `N::Integer`: Total number of observations across all clusters.

Serves as the main container for the simulated dataset used in estimation.
"""
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

"""
    sample(p, Ng)

Draws a full clustered sample from a `clPopulation`.

# Parameters
- `p::clPopulation`: Population object containing the DGP parameters.
- `Ng::Int`: Number of observations per cluster.

# Returns
A `clSample` object containing:
- all simulated clusters,
- the total number of observations `N = p.G * Ng`.

Internally applies `simulateclCluster` to each cluster mean in `p.mu`.
"""
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

"""
    simulateclCluster(mu_g, Ng, β)

Generates a single cluster according to the population DGP.

# Parameters
- `mu_g::Float64`: Cluster-specific mean.
- `Ng::Int64`: Number of observations within the cluster.
- `β::Float64`: True regression coefficient in the structural model.

# Returns
A `clCluster` object containing:
- the regressor vector `Xig`,
- the error vector `uig`,
- the outcome `y_ig`,
- and the associated `DataFrame`.

Internally constructs `Xig` and `uig` via transformations of standard normal draws and builds the implied DGP for `y_ig`.
"""
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

"""
    bols(c)

Computes the OLS estimator using data from a single `clCluster`.

# Parameters
- `c::clCluster`: Cluster containing regressor vector `X` and outcome vector `y`.

# Returns
The scalar OLS coefficient obtained from the cluster-level regression of `y` on `X`.
"""
function bols(c::clCluster)
    (inv(c.X' * c.X) * (c.X' * c.y))
end


"""
    bols(s)

Computes the OLS estimator using the full `clSample`.

# Parameters
- `s::clSample`: Sample composed of multiple clusters.

# Returns
A vector containing the OLS coefficient(s) estimated by stacking all cluster data.

Aggregates all regressors via `getallX(s)` and all outcomes via `getallY(s)` before computing the estimator.
"""
function bols(s::clSample)
    X = getallX(s)
    Y = reduce(vcat, getallY(s))
    betahat = inv(X' * X) * (X'Y)
end



"""
    getX(c)

Returns the regressor vector `X` from a `clCluster`.

# Parameters
- `c::clCluster`: Cluster from which to extract the regressor.

# Returns
A vector containing the cluster’s regressor values.
"""
function getX(c::clCluster)
    c.X
end

"""
    getY(c)

Returns the outcome vector `y` from a `clCluster`.

# Parameters
- `c::clCluster`: Cluster from which to extract the outcome.

# Returns
A vector containing the cluster’s outcome values.
"""
function getY(c::clCluster)
    c.y
end

"""
    getallX(s)

Collects the regressor vectors of all clusters in a `clSample`.

# Parameters
- `s::clSample`: Sample containing multiple clusters.

# Returns
A vector of vectors, where each element is the `X` vector of a cluster.
"""
function getallX(s::clSample)
    reduce(vcat,getX.(s.allclusters)) 
end

"""
    getallY(s)
    
Collects the outcome vectors of all clusters in a `clSample`.

# Parameters
- `s::clSample`: Sample containing multiple clusters.

# Returns
A vector of vectors, where each element is the `y` vector of a cluster.
"""
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
    montecarlo(p, iterations, Ng)

Runs a Monte Carlo experiment using the population specification `p`.

# Parameters
- `p::clPopulation`: Population object defining the DGP.
- `iterations::Int64`: Number of Monte Carlo replications.
- `Ng::Int64`: Number of observations per cluster in each replication.

# Returns
A struct containing as the first element the mean of the OLS estimates produced in each Monte Carlo iteration, and as second the stdev.

Internally replicates the population `p`, draws a sample for each replication, and applies `bols` to each simulated sample.
"""
function montecarlo(p::Population, iterations::Int64, Ng::Int64, allbeta::Bool=false)
    v = fill(p,iterations)
    b = bols.(sample.(v,Ng))
    if allbeta
        return getBeta(b)
    else
        return (mean(getBeta(b)), std(getBeta(b)))
    end
end

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

function innerBCrve(c::clCluster)
    X = c.X
    uhat = computeResidual(c)
    X' * uhat * uhat' * X
end

function CRVE(s::clSample)
    B = sum(innerBCrve.(s.allclusters))
    X = getallX(s)
    inv(X'*X) * B * inv(X'*X)
end

function getBeta(a::Vector{Vector{Float64}})
    permutedims(stack(a))[:,2]
end 