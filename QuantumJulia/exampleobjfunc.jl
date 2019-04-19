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

    verbose = false
    weights = 2
    penalty = 2

	if verbose
   	  objv, grad, pl1, pl2 = objfunc.traceobjgrad(pcof, params, order, verbose, true, weights, penalty)
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

	T = 10.0

	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
	#pcof = rand(4)

	pcof = [1e-3, 2e-3, -2e-3]

#	 m = readdlm("bspline-200-t150.dat")
#	pcof = Array{Float64,1}(m[6:end,1])

#	pcof  = zeros(250)

order = 2

    verbose= true
    adjoint= false
    
	if verbose && adjoint
  	  objv, grad, pl1, pl2 = objfunc.traceobjgrad(pcof,params,order, verbose, adjoint)
        elseif verbose  
  	  objv, pl1, pl2 = objfunc.traceobjgrad(pcof,params,order, verbose, adjoint)
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
	  pl2
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

function testgrad2(pcof0::Array{Float64,1} = [0.,0.,0.,1.,1.,1.])
  N = 2
  Nguard = 0
  Ntot = N + Nguard

  cfl = 0.01
  T = 100
  maxpar =0.09

  # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
  fa = 4.10336
  xia = 2* 0.1099
  samplerate = 64
	
# specify target gate in the rotating frame
  vtarget = zeros(ComplexF64,Ntot,N)

  # -pi/2 y-rot gate pcof0=[0,0,0,1,1,1]
  vtarget[1,1] = 1/sqrt(2)
  vtarget[1,2] = 1/sqrt(2)
  vtarget[2,1] = -1/sqrt(2)
  vtarget[2,2] = 1/sqrt(2)

  # pi/2 x-rot gate: pcof0=[1,1,1,0,0,0]
  # vtarget[1,1] = 1/sqrt(2)
  # vtarget[1,2] = -1im/sqrt(2)
  # vtarget[2,1] = -1im/sqrt(2)
  # vtarget[2,2] = 1/sqrt(2)

  rotmat = [1 0; 0 exp(1im*2*pi*fa*T)]
  utarget = rotmat' * vtarget # add a matching quotation for emacs'
  kpar = 5 # needs to have the same value in traceobjgrad()
  
  params = objfunc.parameters(N,Nguard,T,maxpar,cfl, utarget, fa, xia, samplerate, kpar) 

  absomega = 0.25*pi/T
# scale the input coefficients
  pcof1 = absomega*pcof0
  println("Constant control amplitude: ", absomega)
  
  #	m = readdlm("bsline-file.dat")
  #	pcof1 = Array{Float64,1}(m[6:end,1])

  #  pcfile = "pcof.dat"
  #  println("Reading B-spline coefficients from file '", pcfile, "'")
  #  m = readdlm(pcfile)
  #  pcof1 = Array{Float64,1}(m[1:end,1])
        
  order = 2
  eps = 1e-9
  pcof2 = pcof1

  verbose = true
  weights = 2 # final gate fidelity
  penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
  objv1, grad1, td, labdrive, pl1, pl2  = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
  #    @show(grad1)
  pcof2[kpar] = pcof1[kpar] + eps
  objv2  = objfunc.traceobjgrad(pcof2, params, order, false, false, weights, penaltyweight) # just evaluate the objective fcn

  @show(objv1)
  @show(objv2)
  nmax = min(10,length(grad1))
  @show(nmax)
  @show(grad1[1:nmax])
    
  println("Component kpar = ", kpar, " Gradient: ", grad1[kpar], " Approximated by Finite-Differences: ", (objv2-objv1)/eps)

  return pl1, pl2
end

function testfunc2(pcof0::Array{Float64,1} = [0.,0.,0.,1.,1.,1.])
  N = 2
  Nguard = 0
  Ntot = N + Nguard

  cfl = 0.05
  T = 100
  maxpar =0.09

  # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
  fa = 4.10336
  xia = 2* 0.1099
  samplerate = 64
	
# specify target gate in the rotating frame
  vtarget = zeros(ComplexF64,Ntot,N)

  # -pi/2 y-rot gate pcof0=[0,0,0,1,1,1]
  vtarget[1,1] = 1/sqrt(2)
  vtarget[1,2] = 1/sqrt(2)
  vtarget[2,1] = -1/sqrt(2)
  vtarget[2,2] = 1/sqrt(2)

  # pi/2 x-rot gate pcof0=[1,1,1,0,0,0]
  # vtarget[1,1] = 1/sqrt(2)
  # vtarget[1,2] = -1im/sqrt(2)
  # vtarget[2,1] = -1im/sqrt(2)
  # vtarget[2,2] = 1/sqrt(2)

  rotmat = [1 0; 0 exp(1im*2*pi*fa*T)]
  utarget = rotmat' * vtarget # make emacs happy: '
  
  params = objfunc.parameters(N,Nguard,T,maxpar,cfl, utarget, fa, xia, samplerate) 

  absomega = 0.25*pi/T
# scale the input coefficients
  pcof1 = absomega*pcof0
  println("Constant control amplitude: ", absomega)
  
  #	m = readdlm("bsline-file.dat")
  #	pcof1 = Array{Float64,1}(m[6:end,1])

  #  pcfile = "pcof.dat"
  #  println("Reading B-spline coefficients from file '", pcfile, "'")
  #  m = readdlm(pcfile)
  #  pcof1 = Array{Float64,1}(m[1:end,1])
        
  order = 2
  eps = 1e-9
  kpar = 2 # needs to have the same value in traceobjgrad()
  pcof2 = pcof1

  verbose = true
  weights = 2 # final gate fidelity
  penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
# retun fields for evaladjoiont=false
  objv1, td, labdrive, pl1, pl2 = objfunc.traceobjgrad(pcof1, params, order, verbose, false, weights, penaltyweight)

  @show(objv1)

  # save to file for quantum device (assumes samplerate=32)
  smallnumber = 1e-17
  savevec = zeros(320000) .+ smallnumber
  @show(length(labdrive))
  savevec[1:length(labdrive)] = labdrive

  filename = "control_quantum.dat"
  println("Saving sampled control function for experiment on file '", filename, "'");
  writedlm(filename,savevec)

  # save to file for mesolve in qutip
  filename = "control_qutip.dat"
  println("Saving sampled control function for qutip on file '", filename, "'");
  writedlm(filename, labdrive)

  return pl1, pl2
end

