using Plots
using DelimitedFiles

ta = LinRange(16.0,360.0, 87) # stays the same for all data

dir_a = "vibranium_data/Aug11-25/"
dir_b = "SWAP02_optimize/"
dir_c = "SWAP02_diagonal_UDE/FWD_opt/"

a = readdlm(dir_a * "init_0_pop_cor.dat");
b = readdlm(dir_b * "population0.iinit0000.dat", skipstart=1)
c = readdlm(dir_c * "population0.iinit0000.dat", skipstart=1)

pl0 = plot(legend=:outerright, title="Population starting from |0>")
plot!(xlims=[0,360])
plot!(ta,a[:,1],lab="|0> data",line=:dash, color=:red)
plot!(ta,a[:,2],lab="|1> data",line=:dash, color=:green)
plot!(ta,a[:,3],lab="|2> data",line=:dash, color=:blue)

plot!(b[:,1], b[:,2],lab="|0> ideal",line=:solid, color=:lightsalmon)
plot!(b[:,1], b[:,3],lab="|1> ideal",line=:solid, color=:lightgreen)
plot!(b[:,1], b[:,4],lab="|2> ideal",line=:solid, color=:lightblue)

plot!(c[:,1], c[:,2],lab="|0> tf-opt",line=:solid, color=:red)
plot!(c[:,1], c[:,3],lab="|1> tf-opt",line=:solid, color=:green)
plot!(c[:,1], c[:,4],lab="|2> tf-opt",line=:solid, color=:blue)

println("Plot handle in pl0")

# starting from |1>
a = readdlm(dir_a * "init_1_pop_cor.dat");
b = readdlm(dir_b * "population0.iinit0001.dat", skipstart=1)
c = readdlm(dir_c * "population0.iinit0001.dat", skipstart=1)

pl1 = plot(legend=:outerright, title="Population starting from |1>")
plot!(xlims=[0,360])
plot!(ta,a[:,1],lab="|0> data",line=:dash, color=:red)
plot!(ta,a[:,2],lab="|1> data",line=:dash, color=:green)
plot!(ta,a[:,3],lab="|2> data",line=:dash, color=:blue)

plot!(b[:,1], b[:,2],lab="|0> ideal",line=:solid, color=:lightsalmon)
plot!(b[:,1], b[:,3],lab="|1> ideal",line=:solid, color=:lightgreen)
plot!(b[:,1], b[:,4],lab="|2> ideal",line=:solid, color=:lightblue)

plot!(c[:,1], c[:,2],lab="|0> tf-opt",line=:solid, color=:red)
plot!(c[:,1], c[:,3],lab="|1> tf-opt",line=:solid, color=:green)
plot!(c[:,1], c[:,4],lab="|2> tf-opt",line=:solid, color=:blue)

println("Plot handle in pl1")


# starting from |2>
a = readdlm(dir_a * "init_2_pop_cor.dat");
b = readdlm(dir_b * "population0.iinit0002.dat", skipstart=1)
c = readdlm(dir_c * "population0.iinit0002.dat", skipstart=1)

pl2 = plot(legend=:outerright, title="Population starting from |2>")
plot!(xlims=[0,360])
plot!(ta,a[:,1],lab="|0> data",line=:dash, color=:red)
plot!(ta,a[:,2],lab="|1> data",line=:dash, color=:green)
plot!(ta,a[:,3],lab="|2> data",line=:dash, color=:blue)

plot!(b[:,1], b[:,2],lab="|0> ideal",line=:solid, color=:lightsalmon)
plot!(b[:,1], b[:,3],lab="|1> ideal",line=:solid, color=:lightgreen)
plot!(b[:,1], b[:,4],lab="|2> ideal",line=:solid, color=:lightblue)

plot!(c[:,1], c[:,2],lab="|0> tf-opt",line=:solid, color=:red)
plot!(c[:,1], c[:,3],lab="|1> tf-opt",line=:solid, color=:green)
plot!(c[:,1], c[:,4],lab="|2> tf-opt",line=:solid, color=:blue)

println("Plot handle in pl2")
