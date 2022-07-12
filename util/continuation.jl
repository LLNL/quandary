using LinearAlgebra
using DelimitedFiles

include("utils.jl")

# function write_mat_to_file(A, filename)

#     # Vectorize real and imaginary part and concatenate
#     dim = size(A)[1] * size(A)[1]
#     Re = reshape(real(A), dim, 1)
#     Im = reshape(imag(A), dim, 1)
#     x = [Re;Im]
#     for i = 1:dim*2
#         #println(x[i])
#         if abs(x[i]) < 1e-13
#             x[i] = 0.0
#         end
#     end

#     # Write x to file
#     open(filename, "w") do io
#         writedlm(io, x)
#     end
#     println("File written: "*filename)
# end

function getHamiltonian(U; verbose = false)
    if verbose
        print("Computing H = i*log(U)...   ")
    end
    Hout = im * log(U)
    if verbose
        println("Done.")
    end
    return Hout
end

# Continuation unitary
function Ucont(s, Hsub, Htar)
    Hcont = (1-s)*Hsub + s*Htar
    Uout = exp(-im * Hcont)
    return Uout
end


# dimensions
nlevels = 2
noscils = 5
N = nlevels^noscils

# Traget unitary, here C5NOT
Utar =zeros(Complex{Float64}, N, N)
for i=1:N-2
    Utar[i,i] = 1.0
end
Utar[N-1, N] = 1.0
Utar[N, N-1] = 1.0


# file prefix for suboptimal solution unitary
prefix_re = "rho_Re.iinit"
prefix_im = "rho_Im.iinit"


# Read the current suboptimal unitary from file
print("Reading suboptimal unitary from files "*prefix_re* "[..] and "*prefix_im*"[..] ... ")
Usub =zeros(Complex{Float64}, N, N)
for i=1:N 
    filename_re = prefix_re*lpad(i-1,4,"0")*".dat"
    filename_im = prefix_im*lpad(i-1,4,"0")*".dat"
    state_re = readdlm(filename_re)[2,2:end]
    state_im = readdlm(filename_im)[2,2:end]

    Usub[:,i] = state_re + im*state_im
end
println("Done.")

# Compute suboptimal and target Hamiltonians
print("Computing suboptimal and target Hamiltonians...  ")
Hsub = getHamiltonian(Usub)
Htar = getHamiltonian(Utar)
println("Done.")


# test continuation target at s=0 and s=1
println("Continuation test:")
println("  error at s=0: ", norm(Ucont(0.0, Hsub, Htar) - Usub) )
println("  error at s=1: ", norm(Ucont(1.0, Hsub, Htar) - Utar) )

# Compute and save continuation target at Ns intermediate points
Ns = 20
writemats = false
testunitary = true
ds = 1.0/Ns
for s = 0:Ns
    si = s*ds
    Us = Ucont(si, Hsub, Htar)
    if testunitary
        err1 = norm(Us*Us' - Matrix(I, N, N))
        err2 = norm(Us'*Us - Matrix(I, N, N))
        println("Error = ", 1/2*(err1+err2))
    end
    if writemats
        write_mat_to_file(Us, "Ucont_"*string(round(si, digits=2))*".dat")
    end
end


