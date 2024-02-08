using Plots
using FileIO

function read_binary(filename::String, Ndata::Int64)
    data = Array{Float64}(undef, Ndata)
    read!(filename, data)
    return data
end

pll = plot(xlabel="Index")
for q = 0:4
    # lagrange-outer-0.bin
    local filename = "lagrange-outer-" * string(q) * ".bin"
    Ndata = 512 # 64
    lambda = read_binary(filename, Ndata)
    plot!(lambda,lab="Lambda-"*string(q+1))
end

println("Plot handle in 'pll'")