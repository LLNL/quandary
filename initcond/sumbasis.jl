using LinearAlgebra
using DelimitedFiles

function sumbasis(N, Neye, filename)

    # Construct matrix
    p = (1+im)/2
    m = (1-im)/2

    S =  N * Matrix{Complex{Float64}}(I,N,N)
    
    for i = 1:N
        for j = 1:i-1
            S[i,j] = m
        end
        for j = i+1:N
            S[i,j] = p
        end
    end

    S = S / (N*N)

    # Kronecker S \otimes I
    Id = zeros(Neye,Neye)
    Id[1,1] = 1
    S = kron(S,Id)
    dim = N*N*Neye*Neye

    # Vectorize real and imaginary part and concatenate
    Re = reshape(real(S), dim, 1)
    Im = reshape(imag(S), dim, 1)
    x = [Re;Im]
    for i = 1:dim*2
        println(x[i])
        if abs(x[i]) < 1e-12 
            x[i] = 0.0
        end
    end

    ## Write x to file
    #open(filename, "w") do io
    #    writedlm(io, x)
    #end
end


sumbasis(3,20,"alice_sumbasis.dat")
