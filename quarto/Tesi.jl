#| echo : false
include("../src/ClusterSimulation.jl")
G = 100 #Number of clusters 
Ng = 500 # Number of individuals per cluster
 
β = rand(Normal(2,4),G)

#| echo : false
pop = hePopulation(G, β, randn(G))

#| echo : false
sam = sample(pop, Ng)

#| echo : false
beta2 = 2
μg = rand(Normal(0,4),500)
endoPop = clPopulation(500, μg, beta2)
endoSam = sample(endoPop,100)
S = checkScore(endoSam)

#| echo : false
mean(S), var(S)

#| echo : false
#| label: fig-allclustersconsistency
#| fig-cap: "HIstogram of the difference between the estimated parameters and the true one in all clusters"

histogram(
    permutedims(stack(bols.(sam.allclusters)))[:,2] - β,
    bins = 15
    )

#| echo : false
#| label: fig-clusterwisemontecarlo
#| fig-cap: Histogram of the mean difference between estimated and true parameters across simulation in the montecarlo experiment  
betahatDiff = montecarloClusterWise(pop,100,100) .- β
histogram(mean(betahatDiff, dims = 2), bins =15)

#| output: false
#| echo: false
df = consistencyNg(pop)

#| echo: false
#| label: fig-cons
#| fig-cap: "Consistency of betahat as Ng"
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

#| echo : false
G = 50
β = Normal(2,4)
beta1 = rand(β,G)

#| echo : false
varianceOfBeta = []
for _ in 1:1000
    beta = rand(β,G)
    barBeta = mean(beta)
    push!(varianceOfBeta,barBeta)
end

estimatedVar = (1 / (G - 1)) * sum((beta1 .- mean(beta1)).^2)/G
estimatedStDev = sqrt(estimatedVar)
print("The estimated standard deviation is: " , std(varianceOfBeta), "\nThe montecarlo standard deviation is: ", estimatedStDev)

#| echo : false
v = vcat(collect(1:5:50), collect(50:10:300))
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
    ylabel = "Estimated variance - Montecarlo variance",
    title = "Variance Estimation Error vs Number of Clusters",
    legend = false,
    grid = :on
)

hline!([0.0], linestyle = :dash, linewidth = 1.5)

#| echo : false
iter = 500
Ng = 100
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

#| echo : false
white = getindex.(WhiteAvar.(sam.allclusters),2,2)
print("The average across clusters difference between the estimated variance using white and the Montecarlo variance is: ", mean(white - var(M,dims = 2)))
histogram(white .- var(M, dims = 2), label = "difference between estimated variance and montecarlo one")


vals = round.(Int, exp.(range(log(5), log(300); length = 15)))
vals = unique(vals)

Gs = vals
Ngs = vals


#| echo : false
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

plot1 = plot(
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

#| echo : false
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

plot2 = plot(
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

plot(plot1, plot2; layout = (2, 1), size = (900, 900), dpi = 300)



####### DECOMPOSITION MONTECARLO #############
#| echo : false

crve_white_diff = Dict()
total_var = Dict()
egDict = Dict()
dgDict = Dict()
betaDict = Dict()

for G in Gs
    for Ng in Ngs

        eg_mat = Float64[]
        dg_mat = Float64[]
        betaHatOls = Float64[]
        crve_est = Float64[]
        white_est = Float64[]

        for _ in 1:1000
            beta1 = rand(β,G)
            pop = hePopulation(G,beta1,randn(G))
            sam = sample(pop, Ng)
            beta_hat_g = getBeta(bols.(sam.allclusters))

            betaHat = bols(sam)
            push!(betaHatOls, betaHat[2])

            eg = beta_hat_g .- beta1
            push!(eg_mat, var(eg))

            dg = beta1 .- 2
            push!(dg_mat, mean(dg))

            push!(crve_est, CRVE(sam)[2,2])
            push!(white_est, whiteAvar(sam)[2,2])
        end

        diff = (mean(crve_est) - mean(white_est)) 
        crve_white_diff[(Ng, G)] = diff
        
        dgDict[(Ng,G)] = dg_mat
        egDict[(Ng,G)] = eg_mat
        betaDict[(Ng,G)] = betaHatOls
        total_var[(Ng, G)] = var(eg_mat) + var(dg_mat)
    end
end

var_decomp = Dict()

for G in Gs
    for Ng in Ngs
        b   = betaDict[(Ng, G)]
        egv = egDict[(Ng, G)]
        dgm = dgDict[(Ng, G)]

        lhs = var(b)
        rhs = var(dgm) + mean(egv)

        var_decomp[(Ng, G)] = (
            var_bols        = lhs,
            var_dg_plus_eg  = rhs,
            diff            = lhs - rhs,
            rel_diff        = (lhs - rhs) / lhs
        )
    end
end

plt = plot(xlabel = "Ng", ylabel = "Relative diff", title = "Decomposition error by Ng and G",ms= 5)

for G in sort(Gs)
    rels = [var_decomp[(Ng, G)].rel_diff for Ng in sort(Ngs)]
    plot!(sort(Ngs), rels, marker = :circle, label = "G = $G")
end

display(plt)

using Plots

vd_eq = Dict{Int, NamedTuple}()
for ((Ng, G), v) in var_decomp
    if Ng == G
        vd_eq[G] = (
            var_ols  = v.var_bols,
            var_decp = v.var_dg_plus_eg
        )
    end
end

x = sort(collect(keys(vd_eq)))
var_ols  = [vd_eq[k].var_ols  for k in x]
var_decp = [vd_eq[k].var_decp for k in x]

lo = min.(var_ols, var_decp)
hi = max.(var_ols, var_decp)

p = plot(
    x, var_decp;
    fillrange = 0,
    fillalpha = 0.30,
    linewidth = 2,
    top_margin = 14Plots.mm,
    label = "Our decomposition",
    xscale = :log10,
    dpi = 300
)

plot!(
    p,
    x, var_ols;
    linewidth = 2,
    label = "Monte Carlo Variance"
)

plot!(
    p,
    x, hi;
    fillrange = lo,
    fillalpha = 0.20,
    linewidth = 0,
    label = "Empirical - Estimated Variance"
)

xlabel!(p, "Number of clusters (= cluster size)")
ylabel!(p, "Variance")

subtitle_txt = "When either \$G\$ or \$N_g\$ is large:\n" *
               "• \$d_g\$ and \$e_g\$ are measured very precisely.\n" *
               "• \$\\operatorname{Var}(d_g) + \\mathbb{E}[\\operatorname{Var}(e_g)]\$ closely matches\n" *
               "  \$\\operatorname{Var}(\\hat{\\beta}_{\\mathrm{ols}})\$."

# place using axis-fraction coordinates: (0,0)=bottom-left, (1,1)=top-right
annotate!(
    p,
    (0.02, 1.2),
    text(
        subtitle_txt,
        9,
        :left,
        :top,
        RGB(0, 0, 0)
    );
    annotationcoords = :axes
)

display(p)
savefig(p, "quarto/assets/decompositionOfBetaOLS.pdf")


################################################


#| echo : false
dgVar = Dict{Tuple{Int,Int},Float64}()

for ((Ng, G), dg_vals) in dgDict
    dgVar[(Ng, G)] = var(dg_vals)
end

# Compare diff (CRVE - White) to Var(mean(dg))
results = Dict{Tuple{Int,Int},NamedTuple}()

for key in keys(crve_white_diff)
    diff = crve_white_diff[key]
    vdg  = dgVar[key]

    results[key] = (
        diff              = diff,
        var_mean_dg       = vdg,
        abs_diff          = diff - vdg,
        rel_error         = (diff - vdg) / vdg,
    )
end

# Extract data
Ng_vals = [k[1] for k in keys(results)]
G_vals  = [k[2] for k in keys(results)]
diffs   = [results[k].diff for k in keys(results)]
vdgs    = [results[k].var_mean_dg for k in keys(results)]

scPlot = scatter(
    vdgs,
    diffs;
    marker_z = G_vals,       # COLOR based on G
    color = :turbo,
    xlabel = L"\operatorname{Var}(\bar d_g)",
    ylabel = L"\mathrm{CRVE} - \mathrm{White}",
    title = "CRVE − White vs Var(mean(d_g))",
    ms = 4,
    colorbar_title = L"G",
    legend = false,
    dpi = 300,
)

plot!(identity; lw=2, ls=:dash, label=false)

savefig(scPlot, "./quarto/assets/dgEqualsDiff.pdf")

#| echo : false
#| fig-cap: "Consistency of White Variance estimator without cluster level endogeneity"
#| label: fig-montecarlowhitenoend

betahat = bols(sam)[2]
white = sqrt(whiteAvar(sam)[2,2])
crve = sqrt(CRVE(sam)[2,2])

mc_results = montecarlo(pop, 1000, Ng, true, false)

histogram(mc_results, bins=50, normalize=:pdf, label="Montecarlo")

d = Normal(betahat, white)
D = Normal(betahat, crve)

x = range(betahat - 4 * crve, betahat + 4 * crve; length = 1000)

y = pdf.(d, x)
Y = pdf.(D, x)

plot!(x, y, linewidth = 2, label = "White")
plot!(x, Y, linewidth = 2, label = "CRVE")

#| echo : false
betahat = bols(endoSam)[2]
white = sqrt(whiteAvar(endoSam)[2,2])
crve = sqrt(CRVE(endoSam)[2,2])
histogram(montecarlo(endoPop,400,Ng,true,false), bins=50,normalize=:pdf, label="Montecarlo")
d = Normal(betahat, white)
D = Normal(betahat, crve)

x = range(betahat - 4 * crve, betahat + 4 * crve; length = 1000)
y = pdf.(d,x)
Y = pdf.(D,x)
plot!(x,y, linewidth =2, label = "White")
plot!(x,Y, linewidth =2, label = "CRVE")

#| echo : false
betahat = bols(sam)[2]
white = sqrt(whiteAvar(sam)[2,2])
crve = sqrt(CRVE(sam)[2,2])
histogram(montecarlo(pop,400,Ng,true,true),bins=50,normalize=:pdf, label="Montecarlo")
d = Normal(betahat, white)
D = Normal(betahat, crve)
x = range(betahat - 4 * crve, betahat + 4 * crve; length = 1000)
y = pdf.(d,x)
Y = pdf.(D,x)
plot!(x,y, linewidth =2, label = "White")
plot!(x,Y, linewidth =2, label = "CRVE")

#| echo : false
betahat = bols(endoSam)[2]
white = sqrt(whiteAvar(endoSam)[2,2])
crve = sqrt(CRVE(endoSam)[2,2])
histogram(montecarlo(endoPop,400,Ng,true,true), bins=50,normalize=:pdf, label="Montecarlo")
d = Normal(betahat, white)
D = Normal(betahat, crve)

x = range(betahat - 4 * crve, betahat + 4 * crve; length = 1000)
y = pdf.(d,x)
Y = pdf.(D,x)
plot!(x,y, linewidth =2, label = "White")
plot!(x,Y, linewidth =2, label = "CRVE")
