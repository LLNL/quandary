using LinearAlgebra
using Plots
using FFTW
using DelimitedFiles
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)
gr()
#pyplot()

include("timestep.jl")
include("bsplines.jl")
include("timesteptest.jl")
include("bsplinetest.jl")
include("objfunc.jl")
include("testobjfunc.jl")

