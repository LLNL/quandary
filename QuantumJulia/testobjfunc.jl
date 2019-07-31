#
# testgrad4
#
function testgrad4(pcof0::Array{Float64,1} = [0.,0.,0.,1.,1.,1.],  verbose::Bool = true)
  N = 4
  Nguard = 3
  Ntot = N + Nguard

  # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
  fa = 4.10336
  xia = 2* 0.1099
  samplerate = 64

  Ident = Matrix{Float64}(I, Ntot, Ntot)   
  utarget = Ident[1:Ntot,1:N]
  utarget[:,3] = Ident[:,4]
  utarget[:,4] = Ident[:,3]

  cfl = 0.05
  T = 150
  maxpar =0.09
  kpar = 5 # test this component of the gradient
	
  params = objfunc.parameters(N,Nguard,T,maxpar,cfl, utarget, fa, xia, samplerate, kpar) 

  #	pcof1 = [0.0, 0.0, 0.0]
  #m = readdlm("bspline-200-t150.dat")
  #pcof1 = Array{Float64,1}(m[6:end,1])

  absomega = 0.25*pi/T
# scale the input coefficients
  pcof1 = absomega*pcof0

  order = 2
  eps = 1e-9 # for fd approx of gradient
  pcof2 = pcof1

  weights = 2 # final gate fidelity
  penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
  if verbose
    objv1, grad1, pl1, pl2, td, labdrive = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
  else
    objv1, grad1  = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
  end
  # perturb one element of pcof and only evaluate the objective function
  pcof2[kpar] = pcof1[kpar] + eps
  objv2  = objfunc.traceobjgrad(pcof2, params, order, false, false, weights, penaltyweight)

  @show(objv1)
  @show(objv2)
  npar = min(10,length(grad1))
  @show(grad1[1:npar])
    
  println("Component kpar = ", kpar, " Adjoint gradient: ", grad1[kpar], " Approximated by Finite-Differences: ", (objv2-objv1)/eps)

  if verbose
    return pl1, pl2
  else
    return
  end
end

#
# testgrad2
#
function testgrad2(pcof0::Array{Float64,1} = [0.,0.,0.,1.,1.,1.],   verbose::Bool = true)
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

  kpar = 5 # test this component of the gradient
  
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

  weights = 2 # final gate fidelity
  penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
  if verbose
    objv1, grad1, pl1, pl2, td, labdrive  = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
  else
    objv1, grad1  = objfunc.traceobjgrad(pcof1, params, order, verbose, true, weights, penaltyweight)
  end
  
  pcof2[kpar] = pcof1[kpar] + eps
  #only compute objective function
  objv2  = objfunc.traceobjgrad(pcof2, params, order, false, false, weights, penaltyweight) # just evaluate the objective fcn

  @show(objv1)
  @show(objv2)
  npar = min(10,length(grad1))
  @show(grad1[1:npar])
    
  println("Component kpar = ", kpar, " Adjoint gradient: ", grad1[kpar], " Approximated by Finite-Differences: ", (objv2-objv1)/eps)

  if verbose
    return pl1, pl2
  else
    return
  end
end # end function

#
# testfunc2
#
function testfunc2(pcof0::Array{Float64,1} = [0.,0.,0.,1.,1.,1.], verbose::Bool = true)
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

  weights = 2 # final gate fidelity
  penaltyweight = 2 # constant penalty coefficient, coefficient depends on guard level
    
  # retun fields for evaladjoiont=false
  if verbose
    objv1, pl1, pl2, td, labdrive = objfunc.traceobjgrad(pcof1, params, order, verbose, false, weights, penaltyweight)
  else
    objv1 = objfunc.traceobjgrad(pcof1, params, order, verbose, false, weights, penaltyweight)
  end
  @show(objv1)

  if verbose
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
  else
    return
  end
end

