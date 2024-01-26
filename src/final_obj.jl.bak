function final_obj(pcof0::Array{Float64,1}, p::objparams, verbose::Bool = true)
    
    # shortcut to working_arrays object in p::objparams  
    w = p.wa

    # initializations start here
    alpha = pcof0[1:p.nAlpha] # extract the B-spline-coefficients

    # setup splinepar
    if p.use_bcarrier
        splinepar = bcparams(p.T, p.D1, p.Cfreq, alpha) # Assumes Nunc = 0
    else
        Nsig  = 2*(p.Ncoupled + p.Nunc) # Only uses for regular B-splines
        splinepar = splineparams(p.T, p.D1, Nsig, alpha)
    end

    dt ::Float64 = p.T/p.nsteps # global time step

    if verbose
        println("final_obj: Vector dim Ntot =", p.Ntot , ", Guard levels Nguard = ", p.Nguard , ", Param dim, Psize = ", p.nCoeff, ", Spline coeffs per func, D1= ", p.D1, " Tikhonov coeff: ", p.tik0)
        println("Final time: ", p.T, ", total number of time steps: " , p.nsteps , ", time step: " , dt )
        println("final_obj: length(pcof) =  ", length(pcof0), " nAlpha = ", p.nAlpha, " nWinit = ", p.nWinit)
    end
    
    # Zero out working arrays
    initialize_working_arrays(w)

    # Allocate storage for saving the unitary at the end of each time interval
    Uend_r = Matrix{Float64}(undef, p.Ntot, p.Ntot)
    Uend_i = Matrix{Float64}(undef, p.Ntot, p.Ntot)

    # Total objective
    objf = 0.0
    infid = 0.0
    finalDist = 0.0

    # only consider the final time interval
    interval = p.nTimeIntervals

    if interval == 1
        # initial conditions from Uinit (fixed)
        Winit_r = p.Uinit_r
        Winit_i = p.Uinit_i
    else
        # initial conditions from pcof0 (determined by optimization)
        offc = p.nAlpha + (interval-2)*p.nWinit # for interval = 2 the offset should be nAlpha
        # println("offset 1 = ", offc)
        nMat = p.Ntot^2
        Winit_r = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
        offc += nMat
        # println("offset 2 = ", offc)
        Winit_i = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
    end

    # Evolve the state under Schroedinger's equation
    # NOTE: the S-V scheme treats the real and imaginary parts with different time integrators
    # First compute the solution operator for a basis of real initial conditions: I
    reInitOp = evolve_schroedinger(p, splinepar, p.T0int[interval], p.Uinit_r, p.Uinit_i, p.Tsteps[interval], false)
    
    # Then a basis for purely imaginary initial conditions: iI
    imInitOp = evolve_schroedinger(p, splinepar, p.T0int[interval], p.Uinit_i, p.Uinit_r, p.Tsteps[interval], false)
    
    # Now we can  account for the initial conditions for this time interval and easily calculate the gradient wrt Winit
    # Uend = (reInitop[1] + i*reInitOp[2]) * Winit_r + (imInitOp[1] + i*imInitOp[2]) * Winit_i
    Uend_r = (reInitOp[1] * Winit_r + imInitOp[1] * Winit_i) # real part of above expression
    Uend_i = (reInitOp[2] * Winit_r + imInitOp[2] * Winit_i) # imaginary part

    # final time interval
    infid = abs(1.0-tracefidabs2(Uend_r, Uend_i, p.Utarget_r, p.Utarget_i))
    if p.pFidType == 1 # Frobenius norm^2 of U - V_tg
        finalDist = trace_operator(Uend_r - p.Utarget_r, Uend_r - p.Utarget_r) + trace_operator(Uend_i - p.Utarget_i, Uend_i - p.Utarget_i)
    elseif p.pFidType == 2 # Infidelity
        finalDist = (1.0-tracefidabs2(Uend_r, Uend_i, p.Utarget_r, p.Utarget_i))
    elseif p.pFidType == 3 # Infidelity-squared
        finalDist = (1.0-tracefidabs2(Uend_r, Uend_i, p.Utarget_r, p.Utarget_i))^2
    elseif p.pFidType == 4 # Generalized Infidelity (convex)
        finalDist = (norm(Uend_r)^2 + norm(Uend_i)^2)/p.N - tracefidabs2(Uend_r, Uend_i, p.Utarget_r, p.Utarget_i)
        infid = finalDist # Replace std infidelity by convex one
    end
    objf += finalDist

    # Tikhonov penalty
    tp = tikhonov_pen(alpha, p) 
    objf += tp

    if verbose
        println("Interval # ", interval, " pFidType = ", p.pFidType, " finalDist = ", finalDist, " infid = ", infid)
        println("final_obj():, objf = ", objf, " Tikhonov penalty = ", tp) #, " Total non-unitary penalty term = ", unit_pen) 
        println()
    end

    return objf, infid, tp

end # function final_obj