using Plots
using FileIO
using DelimitedFiles

g = readdlm("grad.dat",header=false)

plg = plot(abs.(g).+1e-20,lab="Grad", xlabel="Index", yscale=:log10)
ylims!(1e-5,1e-2)

println("Plot handle in 'plg'")