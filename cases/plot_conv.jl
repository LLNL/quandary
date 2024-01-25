using Plots
using FileIO
using DelimitedFiles

d, h = readdlm("optim_history.dat",header=true)

pl = plot(d[:,1],d[:,2],lab="Obj",yscale=:log10,xlabel="Iterations")
plot!(d[:,1],d[:,6],lab="Infid",yscale=:log10)
plot!(d[:,1],d[:,3],lab="Norm(grad)",yscale=:log10)
plot!(d[:,1],d[:,7],lab="Tikonov",yscale=:log10)
plot!(d[:,1],d[:,10],lab="Ctrl Energy",yscale=:log10)
plot!(d[:,1],d[:,11].+1e-10,lab="Discont",yscale=:log10)

ylims!(1e-6,30)

println("Plot handle in 'pl'")