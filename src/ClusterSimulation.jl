#LOAD LIBRARIES
using LinearAlgebra
using Statistics
using DataFrames


struct clPopulation
    G::Integer
    mu::Vector{Float64}
    β::Float64
end

struct clCluster
    Ng :: Integer
    df :: DataFrame
    X :: Vector
    y :: Vector
    u :: Vector
end


struct clSample
    allclusters::Array{clCluster,1}
    N::Integer
end


# DATA SIMULLATION
G = 50
Ng = 1000
N = G * Ng
mu_g = randn(G)
β = 2

# s = sqrt.(abs.(mu_g.+10)) #questo 10 qua è v_0
# a = mu_g ./ s
# b = max.(s .^ 2 .- a.^2, 0)
# Z1 = randn(G, Ng)
# Z2 = randn(G, Ng)
# Xig = reshape(s .* Z1, (N,1))
# uig = reshape(a .* Z1 .+ b .* Z2,(N,1))

# y_ig =  Xig .* β .+ uig

# β_ols = (inv(Xig' * Xig) * (Xig' * y_ig))[1][1]

# u_hat = y_ig .- β_ols .*Xig

# Xg = reshape(Xig,Ng,G) #In ogni colonna c'è un cluster 
# yg = reshape(y_ig, Ng, G)

# β_g = mean(diag(inv(Xg' * Xg) * (Xg' * yg)))

function sample()
    df = DataFrame(
        "Xig" => reshape(Xig,N,),
        "y_ig" => reshape(y_ig,N,),
        "uig" => reshape(uig,N,),
        "u_hat" => reshape(u_hat,N,)
        )
end

function sample(p::clPopulation, Ng::Int)
    N = p.G * Ng
    allc = Vector{Float64}(undef, N)
    allc = simulateCluster.(p.mu,Ng, p.β)
end

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

function bols(c::clCluster)
    (inv(c.X' * c.X) * (c.X' * c.y))[1][1]
end

function bols(s::clSample)
    bols.(s.allclusters)
end

s = clPopulation(50, randn(50), 2)

typeof(sample(s,100)[1])
