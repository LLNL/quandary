module objfunc

include("bsplines.jl")
include("timestep.jl")

using LinearAlgebra
using Plots

struct parameters
	N ::Int64		# vector dimension
	Nguard ::Int64 	# number of extra levels
	T 		# final time
	testadjoint #should this really be in param?
	maxpar
	cfl
	utarget


	function parameters(N, Nguard,T, testadjoint, maxpar,cfl)	
	  Ntot = N + Nguard 
	  Ident = Matrix{Float64}(I, Ntot, Ntot)  
 	  utarget = Ident[1:Ntot,1:N]
      utarget[:,3] = Ident[:,4]
      utarget[:,4] = Ident[:,3]	

	  new(N, Nguard, T, testadjoint, maxpar, cfl, utarget)
	end

	function parameters(N, Nguard,T, testadjoint, maxpar,cfl, utarget)		
	  new(N, Nguard, T, testadjoint, maxpar, cfl, utarget)
	end
	#multiple dispatch with extra struct for H0 = 0 ?
end

function traceobjgrad(pcof0 = [0; 0; 0],  params = parameters(4, 3, 150, 1, 0.09, 0.05), order =2, verbose = false, retadjoint = true)
	N = params.N   	
	Nguard = params.Nguard 	
	T = params.T
	testadjoint = params.testadjoint
	labframe = false
	utarget = params.utarget
	cfl = params.cfl
  
 # Parameters used for the gradient
 kpar = 1

	eps = 1e-9
	xi = 1.0/Nguard  	# coef for penalizing forbidden states

	Ntot = N + Nguard
	pcof = pcof0
	D = size(pcof,1)

	# Make sure that each element of pcof is in the prescribed range
	pcof, par1, par0 = boundcof(pcof, D, params.maxpar, eps)

	
	if verbose
    	println("Vector dim Ntot =", Ntot , ", Guard levels Nguard = ", Nguard , ", Param dim D = ", D , ", pcof(1) = ", pcof[1], ", CFL = ", cfl)
  end
 
  # sub-matricesfor the Hamiltonian
  H0 ,amat, adag, omega, domega = rotframematrices(Ntot)
  zeromat = zeros(Ntot,N) 
  
  # patameters for tbsplines
  dtknot = T/(D - 2)
  splineparams = bsplines.splineparams(T, D, D+1, dtknot.*(collect(1:D) .- 1.5), dtknot.*(collect(1:D+1) .- 2), dtknot, pcof)

    	# control functions
  nurbscontrol = 1
  rfunc(t) = bsplines.bspline2(t,splineparams)
  ifunc(t) = 0
  efunc(t) = bsplines.bspline2(t,splineparams)

  if retadjoint
    rfgrad(t) = bsplines.gradbspline2(t,splineparams)
    ifgrad(t) = zeros(length(pcof))
  end

  	# parameters for time integrator
  pcofmax = maximum(abs.(pcof))
  K1 =  pcofmax.*( amat .+  adag)
  lamb = eigvals(K1)
  maxeig2 = maximum(lamb)
  maxeig1 = maximum(abs.(domega))./(2*pi)

  if verbose
		 println("max(d_omega) = ", maximum(abs.(domega)), ", maxeig1 = ", maxeig1,", pcofmax =" ,pcofmax ,", maxeig2 = ", maxeig2)
  end

  maxeig = 0.5(maxeig1 + maxeig2)
  dt = cfl/maxeig
  nsteps = ceil(Int64,T/dt)
  dt = T/nsteps

  gamma, stages = timestep.getgamma(order)

  if verbose
   	println("Final time: ", T, ", number of time steps: " , nsteps , ", time step: " , dt )
  end

    # the basis for the initial data as a matrix
  Ident = Matrix{Float64}(I, Ntot, Ntot)
  U0 = Ident[1:Ntot,1:N]
  W0 = zeromat

  # Split utarget for sormer verlet
  rotmatr(t) = Diagonal(cos.(domega*t))
  rotmati(t) = Diagonal(-sin.(domega*t))

  rotr = Diagonal(cos.(omega*T))
  roti = Diagonal(sin.(omega*T))
 
  vtargetr = rotr*utarget
  vtargeti = roti*utarget
 
  #real and imagianary part of initial condition
  ur = U0
  vi = zeromat

  if verbose
  	usaver = zeros(Ntot,N,nsteps+1)
   	usavei = zeros(Ntot,N,nsteps+1)
   	usaver[:,:,1] = ur
   	usavei[:,:,1] = -vi
  end
    #for computing objfalpha1

  t = 0
  step = 0
  objfv = 0
    
    # Time-dependent matrices for Stromer-Verlet 
  K(t) = H0 + rfunc(t).*(rotmatr(t)*amat + adag*rotmatr(t)') - ifunc(t).*(rotmati(t)*amat + adag*rotmati(t)')
  S(t) = 		  ifunc(t).*(rotmatr(t)*amat - adag*rotmatr(t)') + rfunc(t).*(rotmati(t)*amat - adag*rotmati(t)')

  timestepperforward = timestep.stormerverlet(K,S,Ident)

  wr = zeromat
  wi = zeromat

  objfalpha1 = 0

    # Forward time stepping loop
  for step in 1:nsteps
    #for the objective function
    infidelity0 = weightf(t, T)*(1-tracefidreal(ur, vi, vtargetr, vtargeti, labframe, rotmati(t),rotmatr(t)))
    forbidden0 = xi*penalf(t, T)*normguard(ur, vi, Nguard)

    if retadjoint
      scomplex0 = tracefidcomplex(ur, -vi, vtargetr, vtargeti, labframe, t, omega)
      salpha0 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega)
      forbalpha0 = xi*penalf(t,T)*screal(ur, vi, wr, wi, Nguard)

      rgrad = rfgrad(t)
      igrad = ifgrad(t)

      dar = rotmatr(t)*amat
      dai = rotmati(t)*amat #adag?
      rfalpha = rgrad[kpar]
      ifalpha = igrad[kpar]

      gr0 = rfalpha.*( (dai .-  dai')*ur .- (dar .+ dar')*vi) .+ ifalpha.*(  (dai .+ dai')*vi .+ (dar .-  dar')*ur) #should it really be ifalpha' and ralpha' here?? Different n anders code ..
      gi0 = rfalpha.*( (dar .+  dar')*ur .+ (dai .- dai')*vi) .+ ifalpha.*( -(dai .+ dai')*ur .+ (dar .-  dar')*vi)
    end

    # Stromer-Verlet
    for q in 1:stages
      if retadjoint
       t0=t
       ur0 = ur
       vi0 = vi
      end
       
    	t, ur, vi = timestep.step(timestepperforward, t, ur, vi, dt*gamma[q])

    	infidelity = weightf(t, T)*(1-tracefidreal(ur, vi, vtargetr, vtargeti, labframe,t, omega))
    	forbidden = xi*penalf(t, T)*normguard(ur, vi, Nguard)
    	objfv = objfv + gamma[q]*dt*0.5*(infidelity0 + infidelity + forbidden0 + forbidden)
     	infidelity0 = infidelity
    	forbidden0 = forbidden		

      if retadjoint	
      # Forcing evolving w
       scomplex1 = tracefidcomplex(ur, -vi, vtargetr, vtargeti, labframe, t, omega)
       dar = rotmatr(t)*amat
       dai = rotmati(t)*amat
       rgrad = rfgrad(t)
       igrad = ifgrad(t)
       rfalpha = rgrad[kpar] #Will this return the same va
       ifalpha = igrad[kpar]

       gr1 = rfalpha'.*( (dai .-  dai')*ur .- (dar .+ dar')*vi) .+ ifalpha'.*(  (dai .+ dai')*vi .+ (dar .-  dar')*ur) #should it really be ifalpha' and ralpha' here?? Different n anders code ..
       gi1 = rfalpha'.*( (dar .+  dar')*ur .+ (dai .- dai')*vi) .+ ifalpha'.*( -(dai .+ dai')*ur .+ (dar .-  dar')*vi)

      # Evolve (wr wi)
       temp, wr, wi = timestep.step(timestepperforward, t0, wr, wi, dt*gamma[q], gi0, 0.5*(gr1 + gr0), gi1) 

       salpha1 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega)
       forbalpha1 =  xi*penalf(t,T)*screal(ur, vi, wr, wi, Nguard)   
       objalpha1 = objfalpha1 - gamma[q]*dt*0.5*2.0*real(weightf(t0,T)*conj(scomplex0)*salpha0 + weightf(t,T)*conj(scomplex1)*salpha1) + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1) 
     
       # save previous values for next stage
       scomplex0 = scomplex1
       salpha0 = salpha1
       forbalpha0 = forbalpha1
       gr0 = gr1
       gi0 = gi1
      end  # retadjoint
    end # Stromer-Verlet
    
    if verbose
      usaver[:,:, step + 1] = ur
      usavei[:,:, step + 1] = vi
    end

  end #forward time steppingloop

	ufinalr = rotr'*ur - roti'*vi #should both these matrices me transposed?
	ufinali = -rotr'*vi - roti'*ur 
	ineqpenalty = evalineqpen(pcof, par0, par1);
	objfv = objfv .+ ineqpenalty


  if retadjoint
    dfdp = objfalpha1
    gradobjfadj = zeros(D,1);
    t = T
    dt = -dt
    adiffmax = 0
  
    # terminal conditions for the adjoint state
    lambdar = zeromat
    lambdai = zeromat

    timestepperbackward = timestepperforward
    
    #Backward time stepping loop
    for step in nsteps-1:-1:0
      scomplex0 = tracefidcomplex(ur, -vi, vtargetr, vtargeti, labframe, t, omega)
      sr0 = real(scomplex0)
      si0 = imag(scomplex0)

      hr0 = -weightf(t,T)/N*(sr0*vtargetr + si0*vtargeti)
      hi0 =  weightf(t,T)/N*(sr0*vtargeti - si0*vtargetr)
      hr0[N+1:N+Nguard,:] = xi*penalf(t,T)*ur[N+1:N+Nguard,:]
      hi0[N+1:N+Nguard,:] = xi*penalf(t,T)*vi[N+1:N+Nguard,:]

      # forcing for evolving W (d psi/d alpha1) in the rotating frame
      dar = rotmatr(t)*amat
      dai = rotmati(t)*amat
      rgrad = rfgrad(t)
      igrad = ifgrad(t)

      # separate out contributions from rfgrad and ifgrad (which determine the component of the gradient)
      darr = ( (dai .- dai')*ur .- (dar .+ dar')*vi)
      dari = ( (dar .+ dar')*ur .+ (dai .- dai')*vi)
      dair = ( (dai .+ dai')*vi .+ (dar .- dar')*ur)
      daii = (-(dai .+ dai')*ur .+ (dar .- dar')*vi)
      tr_adjrf = tracefidreal(darr, dari, lambdar, lambdai)
      tr_adjif = tracefidreal(dair, daii, lambdar, lambdai)
      tr_adj0  = rfgrad(t)* tr_adjrf + ifgrad(t)* tr_adjif

        #loop over stages
        for q in 1:stages
          t0 = t
          ur0 = ur
          vi0 = vi

          timestep.step(timestepperforward, t, ur, vi, dt*gamma[q])

          # evolve ur, vi
          t, ur, vi = timestep.step(timestepperbackward, t, ur, vi, dt*gamma[q])

          scomplex1 = tracefidcomplex(ur, -vi, vtargetr, vtargeti, labframe, t, omega)
          sr1 = real(scomplex1)
          si1 = imag(scomplex1)

          hr1 = -weightf(t,T)/N*(sr1*vtargetr + si1*vtargeti)
          hi1 =  weightf(t,T)/N*(sr1*vtargeti - si1*vtargetr)
          hr1[N+1:N+Nguard,:] = xi*penalf(t,T)*ur[N+1:N+Nguard,:]
          hi1[N+1:N+Nguard,:] = xi*penalf(t,T)*vi[N+1:N+Nguard,:]


          # evolve lambdar, lambdai
          temp, lambdar, lambdai = timestep.step(timestepperbackward, t0, lambdar, lambdai, dt*gamma[q], hi0, 0.5*(hr0 + hr1), hi1)

          dar = rotmatr(t)*amat
          dai = rotmati(t)*amat
          rgrad = rfgrad(t)
          igrad = ifgrad(t)
          
          darr = ( (dai .- dai')*ur .- (dar .+ dar')*vi)
          dari = ( (dar .+ dar')*ur .+ (dai .- dai')*vi)
          dair = ( (dai .+ dai')*vi .+ (dar .- dar')*ur)
          daii = (-(dai .+ dai')*ur .+ (dar .- dar')*vi)
          tr_adjrf = tracefidreal(darr, dari, lambdar, lambdai)
          tr_adjif = tracefidreal(dair, daii, lambdar, lambdai)
          tr_adj1  = rfgrad(t)*tr_adjrf + ifgrad(t)*tr_adjif

          # accumulate the gradient of the objective functional
          gradobjfadj = gradobjfadj + gamma[q]*dt*0.5*2.0*(tr_adj0 +  tr_adj1) # dt is negative

          # save for next stage
          scomplex0 = scomplex1
          tr_adj0 = tr_adj1
          hr0 = hr1
          hi0 = hi1
        end 
    end 
  end

  ineqpengrad = evalineqgrad(pcof, par0, par1)


  for k in 1:D
    gradobjfadj[k] = gradobjfadj[k] + ineqpengrad[k]
  end

  dfdp = dfdp + ineqpengrad[kpar]



	if verbose
		println("Inequality penalty: ", ineqpenalty)
		nplot = 1 + nsteps
		println(" Column   Vnrm")
		for q in 1:N
	 		Vnrm = usaver[:,q,nplot]' * usaver[:,q,nplot] + usavei[:,q,nplot]' * usavei[:,q,nplot]
	 		Vnrm = sqrt(Vnrm)
	 		println(q, " | ", Vnrm)
		end

		tplot = range(0, stop = T, length = nplot)
		c = 3 #what is this?
		q = 3

		plt1 = plotunitary(usaver + 1im*usavei,T)

		# Evaluate polynomials at the discrete time levels
		# Evaluate all polynomials on the midpoint grid
		td = collect(range(0, stop = T, length = nsteps +1))

		f1 = plot(td, rfunc(td), lab = "Real", title = "Control function", linewidth = 2)
		f2 = plot(td, efunc(td), title = "Envelope function", linewidth = 2)
		f3 = plot(td, weightf.(td,T), lab = "Gate", title = "Weight functions", linewidth = 2)
		plot!(td, penalf.(td,T), lab = "Forbidden", linewidth = 2)

		plt2 = plot(f1,f2,f3, layout = (3,1))    
	end

  # perhaps not the most pretty construction
	if verbose
     if retadjoint
      return plt1, plt2, objfv, gradobjfadj
     end
    return plt1, plt2, objfv   
  end
 
 if retadjoint
  return objfv, gradobjfadj
 end

 return objfv
end


# returns omega
function omegafun(N)
	omega = zeros(N)
  	omega[1] = 0
  	omega[2] = 4.106
  	omega[3] = 7.992
  	omega[4] = 11.659
  	if N >= 6
  	  omega[5] = 15.105
  	  omega[6] = 18.332
  	end
  	if N >= 7
  	  omega[7] = 21.339
  	end
  	if N > 7
  	  error("not enough frequencies known")
  	end

  	return omega
end

# bound pcof to allowed amplitude
function boundcof(pcof, D, maxpar, eps)
	par1 = maxpar
	par0 = -maxpar

	for q in 1:D
  	  if pcof[q] > par1-eps
  	    pcof[q] = par1-eps
  	  end
  	  if pcof[q] < par0+eps
  	    pcof[q] = par0+eps
  	  end
  	end
	return pcof, par1, par0
end

# Matrices for te hamiltonian in rotation frame
function rotframematrices(Ntot)
    omega = omegafun(Ntot)
	H0 = zeros(Ntot,Ntot)
  	amat = Array(Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U))
  	adag = transpose(amat)
  	domega = zeros(Ntot)
  	domega[1:Ntot-1] = omega[2:Ntot] .- omega[1:Ntot-1]

	return H0, amat, adag, omega, domega
end


function weightf(t, T)
# period
  tp = T/10
  xi = 4/tp # scale factor
  
# center time
  tc = T
  tau = (t - tc)/tp
  mask = (tau >= -0.5) & (tau <= 0.5)
  w = xi*64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3
end

function tracefidreal(ur, vi, vtargetr, vtargeti, labframe,t,omega)
  N = size(vtargetr,2)

  if labframe
  	rotmatc = Diagonal(cos.(omega*t))
  	rotmats = Diagonal(sin.(omega*t))
    ua = rotmatc * ur + rotmats * vi # ur = + Re(u), vi = - Im(u)
    va = rotmats * ur - rotmatc * vi
  else
    ua = ur
    va = -vi
  end
 
  fidelity = (tr(ua' * vtargetr + va' * vtargeti)/N)^2 + (tr(ua' * vtargeti - va' * vtargetr)/N)^2

end

function tracefidreal(frcr, frci, lambdar, lambdai)
  fidreal = tr(frcr' * lambdar + frci' * lambdai);
end

function tracefidcomplex(ur, vi, vtargetr, vtargeti, labframe, t, omega)
  N = size(vtargetr,2)
  fid_cmplx = tr(ur' * vtargetr .+ vi' * vtargeti)/N + 1im*tr(ur' * vtargeti .- vi' * vtargetr)/N;
end

function  penalf(t, T)
  constant = 1.0/T
  alpha = 0
 # period
  tp = T/10
  xi = 4/tp # scale factor for wavelet (integral over half is tp/4)
  # center time
  tc = T
  tau = (t - tc)/tp
  mask = (tau >= -0.5) & (tau <= 0.5)

# weigh the constant and wavelet parts such that max w = xi
  w = alpha * constant + (1-alpha)*xi* 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;
end

function normguard(vr, vi, Nguard)
  Ntot =size(vr,1)
  N = size(vr,2)

  f = 0
  if Nguard > 0
    rguard = vr[N+1:N+Nguard,:] 
    iguard = vi[N+1:N+Nguard,:]
    f = sum(sum(rguard.^2, dims = 2)) + sum(sum(iguard.^2, dims = 2)) # Is this really a good substitute for sumsq? is it even right?
  end

end

function evalineqpen(pcof, par_0, par_1)
  D = size(pcof,1)
  N = size(pcof,2)
  scalef = 0.1
  penalty = zeros(1,N);
  dp2 = par_1 - par_0
  
  for k in 1:D
    penalty[1,:] = penalty[1,:] .- ( log.((pcof[k,:] .- par_0)/dp2) .+ log.((par_1 .- pcof[k,:])/dp2))
  end
  penalty = scalef .* (penalty/D .- 2*log(2))
end

function plotunitary(us, T)
  nsteps = length(us[1,1,:])  
  Ntot = length(us[:,1,1])
  N =  length(us[1,:,1])

  if Ntot != N
    println("INFO plotunitary: Ntot= ", Ntot, " and N = ", N ," are not equal")
  end
  t = range(0, stop = T, length = nsteps)

# one figure for the response of each basis vector
  plotarray = Array{Plots.Plot}(undef, N) #empty array for separate plots

  for ii in 1:N
  		titlestr = string("Response to initial data #", ii)
        h = plot(title = titlestr, size = (1000, 1000))
  		for jj in 1:Ntot
  			labstr = string("State ", jj)
			plot!(t, abs.(us[jj,ii,:]), lab = labstr, linewidth = 4, xlabel = "Time")
		end
		plotarray[ii] = h
  end
  plt = plot(plotarray..., layout = N)
  return plt
end

function evalineqgrad(pcof, par0, par1)
  D = size(pcof,1)
  N = size(pcof,2)
  scalef = 0.1
  pengrad = zeros(D,N)
  for k in 1:D
    pengrad[k,:] = scalef*(1.0./(par1 .- pcof[k,:]) .- 1.0./(pcof[k,:] .- par0))/D
  end
 return pengrad
end

function screal(vr, vi, wr, wi, Nguard)
  Ntot =size(vr,1)
  N = size(vr,2)

  f=0;
  if  Nguard > 0 # should give the last guard much higher weight!
    vrguard = vr[N+1:N+Nguard,:]
    viguard = vi[N+1:N+Nguard,:]
    wrguard = wr[N+1:N+Nguard,:]
    wiguard = wi[N+1:N+Nguard,:]   
    f = tr(vrguard*wrguard') +  tr(viguard*wiguard')
  end

end


end
