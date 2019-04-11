function exampleobjfunc()
	N = 4
	#3
	Nguard = 3
	Ntot = N + Nguard
	
	Ident = Matrix{Float64}(I, Ntot, Ntot)   
	utarget = Ident[1:Ntot,1:N]
	utarget[:,3] = Ident[:,4]
	utarget[:,4] = Ident[:,3]
	
	#utarget[:,1] = Ident[:,2]
	#utarget[:,2] = Ident[:,1]
	
	cfl = 0.05

	T = 150

	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
	#pcof = rand(4)

	pcof = [0.0, 0.0, 0.0]

	# m = readdlm("bspline-200-t150.dat")
	#pcof = Array{Float64,1}(m[6:end,1])
	#pcof  = zeros(250)
	order = 2

    verbose = true
    weights = 2
    penalty = 2

	if verbose
  	    objv, grad,  pl1, pl2 = objfunc.traceobjgrad(pcof, params, order, verbose, true, weights, penalty)
	else
	    objv, grad  = objfunc.traceobjgrad(pcof, params, order, verbose, true, weights, penalty)
	end
	
	println("objv: ", objv)
	println("objgrad: ", grad)
	
	if verbose
	  pl1
	end
end

function example_noguard()
	N = 4
	#3
	Nguard = 0
	Ntot = N + Nguard
	
	Ident = Matrix{Float64}(I, Ntot, Ntot)   
	utarget = Ident[1:Ntot,1:N]
	utarget[:,3] = Ident[:,4]
	utarget[:,4] = Ident[:,3]
	
	#utarget[:,1] = Ident[:,2]
	#utarget[:,2] = Ident[:,1]
	
	cfl = 0.05

	T = 150.0

	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
	#pcof = rand(4)

#	pcof = [1e-3, 2e-3, -2e-3]

#	 m = readdlm("bspline-200-t150.dat")
#	pcof = Array{Float64,1}(m[6:end,1])

	pcof  = zeros(250)

order = 2

    verbose= true
    adjoint= true
    
	if verbose && adjoint
  	     objv, grad, pl1, pl2 = objfunc.traceobjgrad(pcof,params,order, verbose, adjoint)

       elseif verbose  
  	    pl1, pl2, objv = objfunc.traceobjgrad(pcof,params,order, verbose, adjoint)
	elseif adjoint
	    objv, grad  = objfunc.traceobjgrad(pcof, params, order, verbose, adjoint)
	else
	    objv  = objfunc.traceobjgrad(pcof, params, order, verbose, adjoint)
	end
	
	println("objv: ", objv)
	if adjoint
          println("objgrad: ", grad)
	end
	
	if verbose
	  pl1
	end
end


function testgrad()
	N = 4
	Nguard = 3
	Ntot = N + Nguard
	
	Ident = Matrix{Float64}(I, Ntot, Ntot)   
	utarget = Ident[1:Ntot,1:N]
	utarget[:,3] = Ident[:,4]
	utarget[:,4] = Ident[:,3]

	
	cfl = 0.05
	T = 150
	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)

#	pcof1 = [0.0, 0.0, 0.0]
	m = readdlm("bspline-200-t150.dat")
	pcof1 = Array{Float64,1}(m[6:end,1])

	order = 2
	eps = 1e-9
	kpar = 2 # needs to have the same value in traceobjgrad()
	pcof2 = pcof1

    verbose = true
    weights = 2 # final gate fidelity
#    penaltyweight = 1 # time-dependent penalty coefficient, same weight for all guard levels
    penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
    objv1, grad1, pl1, pl2  = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
#    @show(grad1)
    pcof2[kpar] = pcof1[kpar] + eps
    objv2, grad2  = objfunc.traceobjgrad(pcof2, params, order, verbose, true, weights, penaltyweight)

    @show(objv1)
    @show(objv2)
    @show(grad1[1:10])
    @show(grad2[1:10])
    
    println("Component kpar = ", kpar, " Gradient: ", grad1[kpar], " Approximated by Finite-Differences: ", (objv2-objv1)/eps)

    if verbose
       pl2
   end
end

function testgrad2()
	N = 2
	Nguard = 3
	Ntot = N + Nguard
	
	# pi/2 y-rot gate
        utarget = zeros(Ntot,N)
        utarget[1,1] = 1/sqrt(2)
        utarget[1,2] = -1/sqrt(2)
        utarget[2,1] = 1/sqrt(2)
        utarget[2,2] = 1/sqrt(2)

        # CNOT gate
	# Ident = Matrix{Float64}(I, Ntot, Ntot)   
	# utarget = Ident[1:Ntot,1:N]
	# utarget[:,3] = Ident[:,4]
	# utarget[:,4] = Ident[:,3]

	
	cfl = 0.05
	T = 100
	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)

#	pcof1 = [0.0, 0.0, 0.0]
	m = readdlm("bspline-200-t150.dat")
	pcof1 = Array{Float64,1}(m[6:end,1])

	order = 2
	eps = 1e-9
	kpar = 2 # needs to have the same value in traceobjgrad()
	pcof2 = pcof1

    verbose = true
    weights = 2 # final gate fidelity
#    penaltyweight = 1 # time-dependent penalty coefficient, same weight for all guard levels
    penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
    objv1, grad1, pl1, pl2  = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
#    @show(grad1)
    pcof2[kpar] = pcof1[kpar] + eps
    objv2, grad2  = objfunc.traceobjgrad(pcof2, params, order, verbose, true, weights, penaltyweight)

    @show(objv1)
    @show(objv2)
    @show(grad1[1:10])
    @show(grad2[1:10])
    
    println("Component kpar = ", kpar, " Gradient: ", grad1[kpar], " Approximated by Finite-Differences: ", (objv2-objv1)/eps)

    if verbose
       pl2
   end
end
