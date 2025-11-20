#LOAD LIBRARIES
using LinearAlgebra
using Statistics


# DATA SIMULLATION
G = 50
Ng = 1000
N = G * Ng
mu_g = randn(G)
β = 2

s = sqrt.(abs.(mu_g.+10)) #questo 10 qua è v_0
a = mu_g ./ s
b = max.(s .^ 2 .- a.^2, 0)
Z1 = randn(G, Ng)
Z2 = randn(G, Ng)
Xig = reshape(s .* Z1, (N,1))
uig = reshape(a .* Z1 .+ b .* Z2,(N,1))

y_ig =  Xig .* β .+ uig

β_ols = (inv(Xig' * Xig) * (Xig' * y_ig))[1][1]

Xg = reshape(Xig,Ng,G) #In ogni colonna c'è un cluster 
yg = reshape(y_ig, Ng, G)

β_g = mean(diag(inv(Xg' * Xg) * (Xg' * yg)))

