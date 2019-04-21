
using Optim
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

  #	pcof0  = zeros(351) 
  #pcof0 = (rand(250) .- 0.5).*maxpar*0.1
  pcof0  = zeros(6) 

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

  println("Objfunc = ", objv)

  # save to file
  using DelimitedFiles
  Base.show(io::IO, f::Float64) = @printf(io, "%.e", f)
  smallnumber = 1e-17
  savevec = zeros(320000) .+ smallnumber
  @show(length(labdrive))
  savevec[1:length(labdrive)] = labdrive

  filename = "control_quantum.dat"
  println("Saving sampled control function on file '", filename, "'");
  writedlm(filename,savevec)

  # save to file for mesolve in qutip
  filename = "control_qutip.dat"
  println("Saving sampled control function for qutip on file '", filename, "'");
  writedlm(filename, labdrive)

  #save the b-spline coeffs
  bsname = "pcof.dat"
  println("Saving B-spline coefficients on file '", bsname, "'");
  writedlm(bsname, pcof)


	
  	 
	
