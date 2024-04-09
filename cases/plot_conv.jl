using Plots
using FileIO
using DelimitedFiles

d, h = readdlm("optim_history.dat",header=true)

pld = plot(abs.(d[:,12]).+1e-10,lab="Norm^2(disc)",yscale=:log10,xlabel="Iterations", size=[800,300])
ylims!(pld,1e-5,10)

pl = plot(abs.(d[:,2]).+1e-10,lab="|Obj|",yscale=:log10,xlabel="Iterations", legend=:outerright)
plot!(d[:,3],lab="Norm(grad)",yscale=:log10)
plot!(d[:,6],lab="Last Infid",yscale=:log10)

if maximum(abs.(d[:,10])) > 0.0
   plot!(d[:,10],lab="Ctrl Energy",yscale=:log10)
end

if maximum(abs.(d[:,7])) > 0.0
   plot!(d[:,7].+1e-10,lab="Tikhonov",yscale=:log10)
end

if maximum(abs.(d[:,11])) > 0.0
   plot!(abs.(d[:,11]).+1e-10,lab="|Constraint|",yscale=:log10)
end   


ylims!(1e-8,10)

println("Plot handles in 'pl' and 'pld'")