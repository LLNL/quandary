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

	T = 150.0

	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
	#pcof = rand(4)

	pcof = [1e-3, 2e-3, -1e-3,-1e-3]

	order = 2

        verbose= false

	if verbose
  	    pl1, pl2, objv, grad = objfunc.traceobjgrad(pcof,params,order, true, verbose)
	else
	    objv  = objfunc.traceobjgrad(pcof, params, order, verbose, false)
	end
	
	println("objv: ", objv)
	#println("objgrad: ", grad)
	
	if verbose
	  pl1
	end
end