
using Optim
  N = 4
  Nguard = 2
  Ntot = N + Nguard
	
  Ident = Matrix{Float64}(I, Ntot, Ntot)   
  utarget = Ident[1:Ntot,1:N]

  utarget[:,3] = Ident[:,4]
  utarget[:,4] = Ident[:,3]

  cfl = 0.025 # how small does the time step need to be?
  T = 100.0
  maxpar =0.09 # there is a factor 2 in the conversion to the lab frame

  # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
  fa = 4.10336
  xia = 2* 0.1099
  samplerate = 64
	
  kpar = 96 # test this component of the gradient
  
  params = objfunc.parameters(N,Nguard,T,maxpar,cfl, utarget, fa, xia, samplerate, kpar) 

  pcof0  = 0.01*maxpar*rand(250) # initial guess must be even (real + imag parts)
  #pcof0  = zeros(250) # this initial guess converges ok, but gives a different solution

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

  res = optimize(f, gopt!, pcof0, LBFGS(), Optim.Options(show_trace =true, iterations=60))

  @time pcof = Optim.minimizer(res)
  display(res)

  objv, grad, pl1, pl2, td, labdrive = objfunc.traceobjgrad(pcof,params,order, true, true, weight, penaltyweight)

  # println("Objfunc = ", objv)

  # save to file for mesolve in qutip
  filename = "control_qutip.dat"
  println("Saving sampled control function for qutip on file '", filename, "'");
  writedlm(filename, labdrive)

  #save the b-spline coeffs
  bsname = "pcof.dat"
  println("Saving B-spline coefficients on file '", bsname, "'");
  writedlm(bsname, pcof)


	
  	 
	
