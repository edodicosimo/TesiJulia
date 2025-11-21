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

function sample(p::clPopulation, Ng::Int)
    N = p.G * Ng
    allc = Vector{Float64}(undef, N)
    allc = simulateCluster.(p.mu,Ng, p.β)
    sample = clSample(allc, N)
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
    X = vcat(getallX(s)...)
    Y = vcat(getallY(s)...)
    inv(X' * X) * (X'Y)
end

s = clPopulation(50, randn(50), 2)

bols.(sample(s,1000).allclusters)

sample1 = sample(s,1000)

function getX(c::clCluster)
    c.X
end

function getY(c::clCluster)
    c.y
end

function getallX(s::clSample)
    getX.(s.allclusters) #è un Vector{Vector{Float64}}
end

function getallY(s::clSample)
    getY.(s.allclusters) #è un Vector{Vector{Float64}}
end


function montecarlo(p::clPopulation, n::Int64, Ng::Int64)
    v = fill(p,n)
    bols.(sample.(v,Ng))
end