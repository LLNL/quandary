using Plots
using FileIO
using DelimitedFiles

d, h = readdlm("optim_history.dat",header=true)

pl = plot(abs.(d[:,2]).+1e-10,lab="Obj",yscale=:log10,xlabel="Iterations")
plot!(d[:,3],lab="Norm(grad)",yscale=:log10)
plot!(d[:,6],lab="Last Infid",yscale=:log10)

plot!(d[:,10],lab="Ctrl Energy",yscale=:log10)
plot!(d[:,11].+1e-10,lab="Discont",yscale=:log10)

if minimum(d[:,7]) > 0.0
    plot!(d[:,7],lab="Tik",yscale=:log10)
end

ylims!(1e-6,30)

println("Plot handle in 'pl'")