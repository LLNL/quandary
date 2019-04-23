
using Optim
  N = 2
  Nguard = 2
  Ntot = N + Nguard
	
  cfl = 0.01
  T = 100
  maxpar =0.09

  # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
  fa = 4.10336
  fa = 4.103208
  xia = 2* 0.1099
  samplerate = 32

  bsname = "pcof.dat"
  filename = "yrot_320000.dat"
  qfilename = "yrot_32_qutip.dat"
  
# specify target gate in the rotating frame (FOR TESTING AGAINST ANALYTICAL SOL)
  utarget = zeros(ComplexF64,Ntot,N)
  vtarget = zeros(ComplexF64,Ntot,N)

  # # pi/2 y-rot gate pcof0=[0,0,0,1,1,1]
  # vtarget[1,1] = 1/sqrt(2)
  # vtarget[1,2] = -1/sqrt(2)
  # vtarget[2,1] = 1/sqrt(2)
  # vtarget[2,2] = 1/sqrt(2)

  # pi/2 x-rot gate: pcof0=[1,1,1,0,0,0]
  # vtarget[1,1] = 1/sqrt(2)
  # vtarget[1,2] = -1im/sqrt(2)
  # vtarget[2,1] = -1im/sqrt(2)
  # vtarget[2,2] = 1/sqrt(2)

  # rotmat = [1 0; 0 exp(1im*2*pi*fa*T)]
  # utarget[1:2,1:2] = rotmat' * vtarget[1:2,1:2] # add a matching quotation for emacs'

  # pi/2 y-rot gate pcof0=[0,0,0,1,1,1]
  utarget[1,1] = 1/sqrt(2)
  utarget[1,2] = -1/sqrt(2)
  utarget[2,1] = 1/sqrt(2)
  utarget[2,2] = 1/sqrt(2)

  # pi/2 x-rot gate: pcof0=[1,1,1,0,0,0]
  # utarget[1,1] = 1/sqrt(2)
  # utarget[1,2] = -1im/sqrt(2)
  # utarget[2,1] = -1im/sqrt(2)
  # utarget[2,2] = 1/sqrt(2)

  kpar = 5 # test this component of the gradient
  
  params = objfunc.parameters(N,Nguard,T,maxpar,cfl, utarget, fa, xia, samplerate, kpar) 

  #  pcof0  = zeros(6) 
  pcof0  = zeros(30) 
  #pcof0 = maxpar*0.01 * rand(30) 

  order = 2
  weight = 2
  penaltyweight = 2

  function f(pcof)
  #@show(pcof)
    f =objfunc.traceobjgrad(pcof,params,order,false,false,weight,penaltyweight )
    # @show(f)
    return f[1]
  end

  function g!(G,pcof,params,order)
    objf, Gtemp = objfunc.traceobjgrad(pcof,params,order,false, true, weight, penaltyweight )
    	
    Gtemp = vcat(Gtemp...) 
    for i in 1:length(Gtemp)
      G[i] = Gtemp[i]
    end
  end

  gopt!(G,pcof) = g!(G,pcof,params,order)

  res = optimize(f, gopt!, pcof0, LBFGS(),Optim.Options(show_trace =true))

  @time pcof = Optim.minimizer(res)
  display(res)

  objv, grad, pl1, pl2, td, labdrive = objfunc.traceobjgrad(pcof,params,order, true, true, weight, penaltyweight)

  #println("Objfunc = ", objv)

  # save to file
  using DelimitedFiles
  Base.show(io::IO, f::Float64) = @printf(io, "%.e", f)
  smallnumber = 1e-17
  savevec = zeros(320000) .+ smallnumber
  @show(length(labdrive))
  savevec[1:length(labdrive)] = labdrive

  println("Saving sampled control function on file '", filename, "'");
  writedlm(filename,savevec)

  # save to file for mesolve in qutip
  println("Saving sampled control function for qutip on file '", qfilename, "'");
  writedlm(qfilename, labdrive)

  #save the b-spline coeffs
  println("Saving B-spline coefficients on file '", bsname, "'");
  writedlm(bsname, pcof)


	
  	 
	
