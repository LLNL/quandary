using LinearAlgebra
using DelimitedFiles

function create_basis_mats(dim)

    basis_mats = Any[]
   
    # Construct diagonal matrices B_ii 
    for i = 1:dim
        B =  zeros(Complex{Float64},dim,dim)
        B[i,i] = 1.0
        push!(basis_mats, B)
    end
    
    # Construct all other mats B_ij
    for i=1:dim
        for j=1:dim
            if i==j
                continue
            end 
            B = zeros(Complex{Float64},dim,dim)
            B[i,i] = 1.0/2.0
            B[j,j] = 1.0/2.0

            upp = 1.0/2.0
            low = 1.0/2.0
            if j>i  # imaginary basis mat
                upp =  im * upp
                low = -im * low
            end 
            B[i,j] = upp
            B[j,i] = low

            push!(basis_mats, B)
        end
    end

    return basis_mats
end

function sum_basis_mats(dim)
    all = create_basis_mats(dim)
    
    S =  zeros(Complex{Float64},dim,dim)

    for i = 1:length(all)
        S += all[i] 
    end

    S /= dim*dim

    return S
end

function write_mat_to_file(S, filename)

    # Vectorize real and imaginary part and concatenate
    dim = size(S)[1] * size(S)[1]
    Re = reshape(real(S), dim, 1)
    Im = reshape(imag(S), dim, 1)
    x = [Re;Im]
    for i = 1:dim*2
        println(x[i])
        if abs(x[i]) < 1e-12 
            x[i] = 0.0
        end
    end

    # Write x to file
    open(filename, "w") do io
        writedlm(io, x)
    end
end

# N=2: for all possible z-coefficients, tests if the linear combination of basis mats is positive definite
function test_N2_coeffs(coeffs_filename)

    N2_coeffs = readdlm(coeffs_filename)    
    B = create_basis_mats(2)

    test_passed = true
    for i = 1:size(N2_coeffs)[1]
        eig_i = eigvals(transpose(N2_coeffs[i,:])*B)
        if minimum(eigs) < 0
           test_passed=false
        end
    end

    if test_passed
        print("Yay! Test passed: All linear combinations are positive definite\n")
    else
        print("Mehhhh... some eigenvalues are negative!")
    end
end
