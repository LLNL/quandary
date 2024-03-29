using Plots
using FileIO
using DelimitedFiles

function plotcontrol(n::Int64)
  fname = "control" * string(n) * ".dat"
  d, h = readdlm(fname,header=true)

  MHz = 1e3
  
  plc = plot(d[:,1], d[:,2]*MHz, lab="p(t)", xlabel="Time[ns]", ylabel="MHz", size=[900,300])
  plot!(d[:,1], d[:,3]*MHz, lab="q(t)")

  println("Returning plot handle")
  return plc
end