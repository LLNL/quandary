using LinearAlgebra
using DelimitedFiles

function addBkj(S,k,j,N)
    # Diagonal part
    S[k,k] += 1/2.
    S[j,j] += 1/2.

    # Off-diagonal part
    if k<j
        S[k,j] += 1/2.
        S[j,k] += 1/2.
    end
    if j<k
        S[j,k] += im/2.
        S[k,j] -= im/2.
    end
end

function sumbasis(N, filename)

    S = zeros(Complex{Float64}, N, N)

    for k = 1:N
        for j=1:N
            addBkj(S,k,j,N)
        end
    end
    
    S = S / (N*N)

    # Vectorize real and imaginary part and concatenate
    dim = N*N
    Re = reshape(real(S), dim, 1)
    Im = reshape(imag(S), dim, 1)
    x = [Re;Im]
    for i = 1:dim*2
    #    println(x[i])
        if abs(x[i]) < 1e-12 
            x[i] = 0.0
        end
    end

    # Write x to file
    open(filename, "w") do io
        writedlm(io, x)
    end
end


AxC = 3*20
sumbasis(AxC,"AxC_sumFullbasis.dat")
