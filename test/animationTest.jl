frames = [montecarlo(pop,iter,Ng,true,false) for Ng in [10,20,30,40,50,100,500]]

# fix x-axis once
allvals = reduce(vcat, frames)
xmin, xmax = extrema(allvals)

anim = @animate for Ng in eachindex(frames)
    histogram(frames[Ng];
        normalize=:pdf,
        alpha=0.6,
        xlim=(xmin, xmax),
        xlabel=L"\hat{\beta}",
        ylabel="density",
        label=false,
        title="N_g = $Ng"
    )
end

gif(anim, "hist3.gif"; fps=24)