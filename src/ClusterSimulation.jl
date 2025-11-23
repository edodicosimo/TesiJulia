#LOAD LIBRARIES
using LinearAlgebra
using Statistics
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
    X :: Vector
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

Internally applies `simulateCluster` to each cluster mean in `p.mu`.
"""
function sample(p::clPopulation, Ng::Int)
    N = p.G * Ng
    allc = simulateCluster.(p.mu,Ng, p.β)
    sample = clSample(allc, N)
end

function sample(p::hePopulation, Ng::Int)
    N = p.G * Ng

end

"""
    simulateCluster(mu_g, Ng, β)

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
function simulateCluster(mu_g::Float64, Ng::Int64, β::Float64)
    s = sqrt.(abs.(mu_g.+10)) #questo 10 qua è v_0
    a = mu_g ./ s
    b = max.(s .^ 2 .- a.^2, 0)
    Z1 = randn(Ng)
    Z2 = randn(Ng)
    Xig = s .* Z1
    uig = a .* Z1 .+ b .* Z2
    y_ig =  Xig .* β .+ uig
    df = DataFrame(
        "Xig" => Xig,
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
    (inv(c.X' * c.X) * (c.X' * c.y))[1][1]
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
    X = vcat(getallX(s)...)
    Y = vcat(getallY(s)...)
    inv(X' * X) * (X'Y)
end

s = clPopulation(50, randn(50), 2)

bols.(sample(s,1000).allclusters)

sample1 = sample(s,1000)

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
    getX.(s.allclusters) 
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

### SI POTREBBE FARE UN STRUCT MONTECARLO, MA PER ORA è IMPLEMENTATO COME UN SEMPLICE VETTORE QUINDI CI STA
"""
    montecarlo(p, iterations, Ng)

Runs a Monte Carlo experiment using the population specification `p`.

# Parameters
- `p::clPopulation`: Population object defining the DGP.
- `iterations::Int64`: Number of Monte Carlo replications.
- `Ng::Int64`: Number of observations per cluster in each replication.

# Returns
A vector containing the OLS estimates produced in each Monte Carlo iteration.

Internally replicates the population `p`, draws a sample for each replication, and applies `bols` to each simulated sample.
"""
function montecarlo(p::clPopulation, iterations::Int64, Ng::Int64)
    v = fill(p,iterations)
    bols.(sample.(v,Ng))
end

