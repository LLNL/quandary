module objfunc

include("bsplines.jl")
include("timestep.jl")

using LinearAlgebra
using Plots

struct parameters
	N ::Int64		# vector dimension
	Nguard ::Int64 	# number of extra levels
	T::Float64		# final time
	testadjoint::Bool #should this really be in param?
	maxpar::Float64
	cfl::Float64
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

function traceobjgrad(pcof0::Array{Float64,1} = [0.0; 0.0; 0.0],  params::parameters = parameters(4, 3, 150, 1, 0.09, 0.05), order::Int64 =2, verbose::Bool = false, retadjoint::Bool = true)
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

  if retadjoint
    @inline rfgrad(t::Float64) = bsplines.gradbspline2(t,splineparams)
    @inline ifgrad(t::Float64) = zeros(length(pcof))
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

  rotr = Diagonal(cos.(omega*T))
  roti = Diagonal(sin.(omega*T))
 
  vtargetr = rotr*utarget
  vtargeti = roti*utarget
 
  #real and imagianary part of initial condition
  vr = U0
  vi = zeromat

  if verbose
  	usaver = zeros(Ntot,N,nsteps+1)
   	usavei = zeros(Ntot,N,nsteps+1)
   	usaver[:,:,1] = vr
   	usavei[:,:,1] = -vi
  end
    #for computing objfalpha1

  t = 0.0
  step = 0
  objfv = 0.0

  Kstep(t::Float64) = K(t,amat,adag,domega,splineparams,H0)
  Sstep(t::Float64) = S(t,amat,adag,domega,splineparams)   
  timestepperforward = timestep.stormerverlet(Kstep,Sstep,Ident)

  wr = zeromat
  wi = zeromat

  objf_alpha1 = 0.0


    # Forward time stepping loop
  for step in 1:nsteps
    #for the objective function
    infidelity0 = weightf(t, T)*(1-tracefidreal(vr, vi, vtargetr, vtargeti, labframe, t, omega))
    forbidden0 = xi*penalf(t, T)*normguard(vr, vi, Nguard)

    if retadjoint
      scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
      salpha0 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega)
      forbalpha0 = xi*penalf(t,T)*screal(vr, vi, wr, wi, Nguard)

      rgrad = rfgrad(t)
      igrad = ifgrad(t)

      dar = rotmatr(t,domega)*amat
      dai = rotmati(t,domega)*amat #adag?
      rfalpha = rgrad[kpar]
      ifalpha = igrad[kpar]

      gr0 = rfalpha.*( (dai .-  dai')*vr .- (dar .+ dar')*vi) .+ ifalpha.*(  (dai .+ dai')*vi .+ (dar .-  dar')*vr) #should it really be ifalpha' and ralpha' here?? Different n anders code ..
      gi0 = rfalpha.*( (dar .+  dar')*vr .+ (dai .- dai')*vi) .+ ifalpha.*( -(dai .+ dai')*vr .+ (dar .-  dar')*vi)
    end

    # Stromer-Verlet
    for q in 1:stages
      if retadjoint
       t0=t
       vr0 = vr
       vi0 = vi
      end
       
    	@inbounds t, vr, vi = timestep.step(timestepperforward, t, vr, vi, dt*gamma[q])

    	infidelity = weightf(t, T)*(1-tracefidreal(vr, vi, vtargetr, vtargeti, labframe,t, omega))
    	forbidden = xi*penalf(t, T)*normguard(vr, vi, Nguard)
    	objfv = objfv + gamma[q]*dt*0.5*(infidelity0 + infidelity + forbidden0 + forbidden)
     	infidelity0 = infidelity
    	forbidden0 = forbidden		

      if retadjoint	
      # Forcing evolving w
       scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
       dar = rotmatr(t,domega)*amat
       dai = rotmati(t,domega)*amat
       rgrad = rfgrad(t)
       igrad = ifgrad(t)
       rfalpha = rgrad[kpar] #Will this return the same va
       ifalpha = igrad[kpar]

       gr1 = rfalpha'.*( (dai .-  dai')*vr .- (dar .+ dar')*vi) .+ ifalpha'.*(  (dai .+ dai')*vi .+ (dar .-  dar')*vr) #should it really be ifalpha' and ralpha' here?? Different n anders code ..
       gi1 = rfalpha'.*( (dar .+  dar')*vr .+ (dai .- dai')*vi) .+ ifalpha'.*( -(dai .+ dai')*vr .+ (dar .-  dar')*vi)

       @inbounds temp, wr, wi = timestep.step(timestepperforward, t0, wr, wi, dt*gamma[q], gi0, 0.5*(gr1 + gr0), gi1) 

       salpha1 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega)
       forbalpha1 =  xi*penalf(t,T)*screal(vr, vi, wr, wi, Nguard)   
       objf_alpha1 = objf_alpha1 - gamma[q]*dt*0.5*2.0*real(weightf(t0,T)*conj(scomplex0)*salpha0 +
          weightf(t,T)*conj(scomplex1)*salpha1) + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1)
#       @show(step, scomplex0, salpha0, scomplex1, salpha1, forbalpha0 , forbalpha1, weightf(t0,T), weightf(t,T))
     
       # save previous values for next stage
       scomplex0 = scomplex1
       salpha0 = salpha1
       forbalpha0 = forbalpha1
       gr0 = gr1
       gi0 = gi1
      end  # retadjoint
    end # Stromer-Verlet
    
    if verbose
      usaver[:,:, step + 1] = vr
      usavei[:,:, step + 1] = -vi
    end

  end #forward time steppingloop

	ufinalr = rotr'*vr - roti'*vi #should both these matrices be transposed?
	ufinali = -rotr'*vi - roti'*vr 
	ineqpenalty = evalineqpen(pcof, par0, par1);
	objfv = objfv .+ ineqpenalty


  if retadjoint
     println("objf_alpha1 = ", objf_alpha1)
    dfdp = objf_alpha1
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
      scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
      sr0 = real(scomplex0)
      si0 = imag(scomplex0)

      hr0 = -weightf(t,T)/N*(sr0*vtargetr + si0*vtargeti)
      hi0 =  weightf(t,T)/N*(sr0*vtargeti - si0*vtargetr)
      hr0[N+1:N+Nguard,:] = xi*penalf(t,T)*vr[N+1:N+Nguard,:]
      hi0[N+1:N+Nguard,:] = xi*penalf(t,T)*vi[N+1:N+Nguard,:]

      # forcing for evolving W (d psi/d alpha1) in the rotating frame
      dar = rotmatr(t,domega)*amat
      dai = rotmati(t,domega)*amat
      rgrad = rfgrad(t)
      igrad = ifgrad(t)

      # separate out contributions from rfgrad and ifgrad (which determine the component of the gradient)
      darr = ( (dai .- dai')*vr .- (dar .+ dar')*vi)
      dari = ( (dar .+ dar')*vr .+ (dai .- dai')*vi)
      dair = ( (dai .+ dai')*vi .+ (dar .- dar')*vr)
      daii = (-(dai .+ dai')*vr .+ (dar .- dar')*vi)
      tr_adjrf = tracefidreal(darr, dari, lambdar, lambdai)
      tr_adjif = tracefidreal(dair, daii, lambdar, lambdai)
      tr_adj0  = rfgrad(t)* tr_adjrf + ifgrad(t)* tr_adjif

        #loop over stages
        for q in 1:stages
          t0 = t
          vr0 = vr
          vi0 = vi

          # evolve vr, vi
          @inbounds t, vr, vi = timestep.step(timestepperbackward, t, vr, vi, dt*gamma[q])

          scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
          sr1 = real(scomplex1)
          si1 = imag(scomplex1)

          hr1 = -weightf(t,T)/N*(sr1*vtargetr + si1*vtargeti)
          hi1 =  weightf(t,T)/N*(sr1*vtargeti - si1*vtargetr)
          hr1[N+1:N+Nguard,:] = xi*penalf(t,T)*vr[N+1:N+Nguard,:]
          hi1[N+1:N+Nguard,:] = xi*penalf(t,T)*vi[N+1:N+Nguard,:]


          # evolve lambdar, lambdai
          @inbounds temp, lambdar, lambdai = timestep.step(timestepperbackward, t0, lambdar, lambdai, dt*gamma[q], hi0, 0.5*(hr0 + hr1), hi1)

          dar = rotmatr(t,domega)*amat
          dai = rotmati(t,domega)*amat
          rgrad = rfgrad(t)
          igrad = ifgrad(t)
          
          darr = ( (dai .- dai')*vr .- (dar .+ dar')*vi)
          dari = ( (dar .+ dar')*vr .+ (dai .- dai')*vi)
          dair = ( (dai .+ dai')*vi .+ (dar .- dar')*vr)
          daii = (-(dai .+ dai')*vr .+ (dar .- dar')*vi)
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
 
    ineqpengrad = evalineqgrad(pcof, par0, par1)

  
    for k in 1:D
      gradobjfadj[k] = gradobjfadj[k] + ineqpengrad[k]
    end
  end

	if verbose
		println("Inequality penalty: ", ineqpenalty)
		dfdp = dfdp + ineqpengrad[kpar]
		println("Forward integration of gradient of objective function = ", dfdp, " ineqpengrad = ", ineqpengrad[kpar])
		
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

    rplot(t) = rfunc(t,splineparams)
    eplot(t) = efunc(t,splineparams)

		f1 = plot(td, rplot.(collect(td)), lab = "Real", title = "Control function", linewidth = 2)
		f2 = plot(td, eplot.(collect(td)), title = "Envelope function", linewidth = 2)
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
function omegafun(N::Int64)
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
function boundcof(pcof::Array{Float64,1}, D::Int64, maxpar::Float64, eps::Float64)
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
function rotframematrices(Ntot::Int64)
    omega = omegafun(Ntot)
	H0 = zeros(Ntot,Ntot)
  	amat = Array(Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U))
  	adag = Array(transpose(amat))
  	domega = zeros(Ntot)
  	domega[1:Ntot-1] = omega[2:Ntot] .- omega[1:Ntot-1]

	return H0, amat, adag, omega, domega
end


function weightf(t::Float64, T::Float64)
# period
  tp = T/10
  xi = 4/tp # scale factor
  
# center time
  tc = T
  tau = (t - tc)/tp
  mask = (tau >= -0.5) & (tau <= 0.5)
  w = xi*64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3
end

function tracefidreal(ur::Array{Float64,2}, vi::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2}, labframe::Bool,t::Float64,omega::Array{Float64,1})
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

function tracefidreal(frcr::Array{Float64,2}, frci::Array{Float64,2}, lambdar::Array{Float64,2}, lambdai::Array{Float64,2})
  fidreal = 0.0
  fidreal = tr(frcr' * lambdar + frci' * lambdai);
end

function tracefidcomplex(ur::Array{Float64,2}, vi::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2}, labframe::Bool, t::Float64, omega::Array{Float64,1})
  N = size(vtargetr,2)
  fidreal = 0.0
  fid_cmplx = tr(ur' * vtargetr .+ vi' * vtargeti)/N + 1im*tr(ur' * vtargeti .- vi' * vtargetr)/N;
end

function  penalf(t::Float64, T::Float64)
  w = 0.0
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

function normguard(vr::Array{Float64,2}, vi::Array{Float64,2}, Nguard::Int64)
  Ntot =size(vr,1)
  N = size(vr,2)

  f = 0.0
  if Nguard > 0
    rguard = vr[N+1:N+Nguard,:] 
    iguard = vi[N+1:N+Nguard,:]
    f = sum(sum(rguard.^2, dims = 2)) + sum(sum(iguard.^2, dims = 2)) # Is this really a good substitute for sumsq? is it even right?
  end

end

function evalineqpen(pcof::Array{Float64,1}, par_0::Float64, par_1::Float64)
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

function evalineqgrad(pcof::Array{Float64,1}, par0::Float64, par1::Float64)
  D = size(pcof,1)
  N = size(pcof,2)
  scalef = 0.1
  pengrad = zeros(D,N)
  for k in 1:D
    pengrad[k,:] = scalef*(1.0./(par1 .- pcof[k,:]) .- 1.0./(pcof[k,:] .- par0))/D
  end
 return pengrad
end

function screal(vr::Array{Float64,2}, vi::Array{Float64,2}, wr::Array{Float64,2}, wi::Array{Float64,2}, Nguard::Int64)
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

@inline function K(t::Float64,amat::Array{Float64,2},adag::Array{Float64,2},domega::Array{Float64,1},splineparams::bsplines.splineparams,H0::Array{Float64,2})
 K = zeros(size(amat)) # To preallocate K, is this nessesary?
 rr = rotmatr(t,domega)
 ri = rotmati(t,domega)
 K = H0 + rfunc(t,splineparams).*(rr*amat + adag*rr') - ifunc(t,splineparams).*(ri*amat + adag*ri)
end

@inline function S(t::Float64,amat::Array{Float64,2},adag::Array{Float64,2},domega::Array{Float64,1},splineparams::bsplines.splineparams)
  S  = zeros(size(amat))
  rr = rotmatr(t,domega)
  ri = rotmati(t,domega)
  S  = ifunc(t,splineparams).*(rr*amat - adag*rr') + rfunc(t,splineparams).*(ri*amat - adag*ri')
end

@inline function rfunc(t::Float64,splineparams::bsplines.splineparams)
  ret = 0.0
  ret =  bsplines.bspline2(t,splineparams)
end

@inline function efunc(t::Float64,splineparams::bsplines.splineparams)
  ret = 0.0
  ret = bsplines.bspline2(t,splineparams::bsplines.splineparams)
end

@inline function ifunc(t::Float64,splineparams::bsplines.splineparams)
  ret = 0.0
end

@inline function rotmatr(t::Float64,domega::Array{Float64,1})
  N = length(domega)
  rotmatr =zeros(N,N)
  rotmatr = Diagonal(cos.(domega*t))
end

@inline function rotmati(t::Float64,domega::Array{Float64,1})
  N = length(domega)
  rotmatr =zeros(N,N)
  rotmatr = Diagonal(-sin.(domega*t))
end


end
