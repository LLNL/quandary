module objfunc

include("bsplines.jl")
include("timestep.jl")

using LinearAlgebra
using Plots
using Printf
using LaTeXStrings

struct parameters
	N::Int64		# vector dimension
	Nguard ::Int64 	# number of extra levels
	T::Float64		# final time
	testadjoint::Bool #should this really be in param?
	maxpar::Float64
	cfl::Float64
	utarget
        samplerate::Int64 # plotting sample rate per ns


	function parameters(N, Nguard,T, testadjoint, maxpar,cfl)	
	  Ntot = N + Nguard 
	  Ident = Matrix{Float64}(I, Ntot, Ntot)  
# CNOT gate assumes N=4
 	  utarget = Ident[1:Ntot,1:N]
          utarget[:,3] = Ident[:,4]
          utarget[:,4] = Ident[:,3]	
          samplerate = 32
	  new(N, Nguard, T, testadjoint, maxpar, cfl, utarget, samplerate)
	end

	function parameters(N, Nguard,T, testadjoint, maxpar,cfl, utarget)		
          samplerate = 32
	  new(N, Nguard, T, testadjoint, maxpar, cfl, utarget, samplerate)
	end
	#multiple dispatch with extra struct for H0 = 0 ?
end

function traceobjgrad(pcof0::Array{Float64,1} = [0.0; 0.0; 0.0],  params::parameters = parameters(4, 3, 150, 1, 0.09, 0.05), order::Int64 =2, verbose::Bool = false, retadjoint::Bool = true, weight = 2, penaltyweight = 2)  
  N = params.N    
  Nguard = params.Nguard  
  T = params.T
  testadjoint = params.testadjoint
  labframe = false
  utarget = params.utarget
  cfl = params.cfl
  samplerate = params.samplerate
  
  tinv = 1.0/T
 # Parameters used for the gradient
  kpar = 2
  eps = 1e-9
  Ntot = N + Nguard
  pcof = pcof0
  D = size(pcof,1)

  if weight == 1
   xif = 1.0
  elseif weight == 2
   xif = 1/T
  end 

  if penaltyweight == 1
    xi = xif/max(1,Nguard)          # coef for penalizing forbidden states
  # elseif penaltyweight == 2  
  #   xi = xif/max(1,Nguard)   
  end
  # Make sure that each element of pcof is in the prescribed range
  pcof, par1, par0 = boundcof(pcof, D, params.maxpar, eps)
      
  if verbose
    @show(utarget)
    println("Vector dim Ntot =", Ntot , ", Guard levels Nguard = ", Nguard , ", Param dim D = ", D , ", pcof(1) = ", pcof[1], ", CFL = ", cfl, " penaltystyle = ", penaltyweight, " samplerate = ", samplerate)
  end
 
  # sub-matricesfor the Hamiltonian
  H0 ,amat, adag, omega, domega = rotframematrices(Ntot)
  zeromat = zeros(Ntot,N) 

# coefficients for penaty style #2 (wmat is a Diagonal matrix)
  wmat = wmatsetup(Ntot, Nguard)
  if verbose && Nguard>0
    @show(wmat[N+1:Ntot, N+1:Ntot])
  end

  # parameters for tbsplines
  dtknot = T/(D - 2)
  splineparams = bsplines.splineparams(T, D, D+1, dtknot.*(collect(1:D) .- 1.5), dtknot.*(collect(1:D+1) .- 2), dtknot, pcof)

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
 
  vtargetr = real((rotr + 1im*roti)*utarget)
  vtargeti = imag((rotr + 1im*roti)*utarget)
 
  #real and imagianary part of initial condition
  vr = U0
  vi = zeromat

  if verbose
  	usaver = zeros(Ntot,N,nsteps+1)
   	usavei = zeros(Ntot,N,nsteps+1)
   	usaver[:,:,1] = vr
   	usavei[:,:,1] = -vi

    #to compute gradient with forward method
    if retadjoint
      wr = zeromat
      wi = zeromat
      objf_alpha1 = 0.0
    end
  end

  # Preallocate
  K0= zeros(Ntot,Ntot)
  S0= zeros(Ntot,Ntot)
  tmp5= zeros(Ntot,Ntot)
  K05= zeros(Ntot,Ntot)
  S05 = zeros(Ntot,Ntot)
  K1 = zeros(Ntot,Ntot)
  S1 = zeros(Ntot,Ntot)
  dar = zeros(Ntot,Ntot)
  dai = zeros(Ntot,Ntot)
  tmp1 = zeros(Ntot,Ntot)
  tmp2 = zeros(Ntot,Ntot)
  tmp3 = zeros(Ntot,Ntot)
  rr = zeros(Ntot,Ntot)
  ri = zeros(Ntot,Ntot)

  gr0 = zeromat
  gi0 = zeromat
  gr1 = zeromat
  gi1 = zeromat
  hr0 = zeromat
  hi0 = zeromat
  hr1 = zeromat
  hi1 = zeromat
  darr = zeromat
  dari = zeromat
  dair = zeromat
  daii = zeromat

  #initialize variables for time stepping
  t = 0.0
  step = 0
  objfv = 0.0

  # Forward time stepping loop
  for step in 1:nsteps
    #for the objective function
    if weight == 1
      infidelity0 = weightf1(t, T)*(1-tracefidreal(vr, vi, vtargetr, vtargeti, labframe, t, omega))
    end

    if penaltyweight == 1
      forbidden0 = xi*penalf1(t, T)*normguard(vr, vi, Nguard)
    elseif penaltyweight == 2
      forbidden0 = tinv*penalf2a(vr, vi, wmat)  
    end

    rotmatrices!(t,domega,rr,ri)

    if retadjoint
      if verbose
       if penaltyweight == 1
         forbalpha0 = xi*penalf1(t,T)*screal(vr, vi, wr, wi, Nguard,zeromat)
       else
        forbalpha0 = tinv*penalf2grad(vr, vi, wr, wi, wmat)
       end
      end

      if weight ==1
        scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)

        if verbose
          salpha0 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega)
        end

        rgrad = rfgrad(t)
        igrad = ifgrad(t)

        mul!(dar,rr,amat)
        mul!(dai,ri,amat)

        rfalpha = rgrad[kpar]
        ifalpha = igrad[kpar]
        gr0, gi0 = grupdate(gr0,gi0,dai, dar, vr, vi, rfalpha, ifalpha)
      end
    end

    # Stromer-Verlet
    for q in 1:stages
      if retadjoint && verbose
       t0=t
       vr0 = vr
       vi0 = vi
      end
      # Update K and S
      KS!(K0, S0, t, amat, adag, domega, splineparams, H0, tmp1, tmp2, tmp3, rr,ri)
      rotmatrices!(t + 0.5*dt*gamma[q],domega,rr,ri)
      KS!(K05, S05, t + 0.5*dt*gamma[q], amat, adag, domega, splineparams, H0, tmp1, tmp2, tmp3,rr,ri)
      rotmatrices!(t + dt*gamma[q],domega,rr,ri)
      KS!(K1, S1, t + dt*gamma[q], amat, adag, domega, splineparams, H0, tmp1, tmp2, tmp3, rr,ri)

      @inbounds t, vr, vi = timestep.step(t, vr, vi, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident)

      if penaltyweight == 1
        forbidden = xi*penalf1(t, T)*normguard(vr, vi, Nguard)  
      elseif penaltyweight == 2
        forbidden = tinv*penalf2a(vr, vi, wmat)  
      end

      if weight == 1
      	infidelity = weightf1(t, T)*(1-tracefidreal(vr, vi, vtargetr, vtargeti, labframe,t, omega))
      	objfv = objfv + gamma[q]*dt*0.5*(infidelity0 + infidelity + forbidden0 + forbidden)
        infidelity0 = infidelity
      elseif weight == 2
        objfv = objfv + gamma[q]*dt*0.5*(forbidden0 + forbidden)
      end 		

      forbidden0 = forbidden

      # compute component of the gradient for verification of adjoint method
      if retadjoint && verbose       	
      	 scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
      	 mul!(dar,rr,amat)
      	 mul!(dai,ri,amat)
      	 rgrad = rfgrad(t)
      	 igrad = ifgrad(t)
      	 rfalpha = rgrad[kpar] #Will this return the same va
      	 ifalpha = igrad[kpar]
	     gr1, gi1 = grupdate(gr1 ,gi1, dai, dar, vr, vi, rfalpha, ifalpha)

	    @inbounds temp, wr, wi = timestep.step(t0, wr, wi, dt*gamma[q], gi0, 0.5*(gr1 + gr0), gi1, K0, S0, K05, S05, K1, S1, Ident) 

	    # Forcing evolving w
      if penaltyweight == 1
        forbalpha1 =  xi*penalf1(t,T)*screal(vr, vi, wr, wi, Nguard,zeromat)
      elseif penaltyweight == 2
        forbalpha1 = tinv*penalf2grad(vr, vi, wr, wi, wmat)
      end

      if weight == 1
        salpha1 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega) 
	objf_alpha1 = objf_alpha1 - gamma[q]*dt*0.5*2.0*real(weightf1(t0,T)*conj(scomplex0)*salpha0 + weightf1(t,T)*conj(scomplex1)*salpha1) + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1)
	scomplex0 = scomplex1
	salpha0 = salpha1
      elseif weight ==2
        objf_alpha1 = objf_alpha1 + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1)
      end
       
      # save previous values for next stage 
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

  if weight == 2
    objfv = objfv + (1-tracefidreal(vr, vi, vtargetr, vtargeti, labframe,t, omega))
    if retadjoint && verbose
      salpha1 = tracefidcomplex(wr, -wi, vtargetr, vtargeti, labframe, t, omega)
      scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
      objf_alpha1 = objf_alpha1 - real(2*conj(scomplex1)*salpha1)
    end  
  end

  vfinalr = vr
  vfinali = -vi

  ufinalr = rotr'*vr - roti'*vi #should both these matrices be transposed?
  ufinali = -rotr'*vi - roti'*vr 
  ineqpenalty = evalineqpen(pcof, par0, par1);
  objfv = objfv .+ ineqpenalty

  if retadjoint
    if verbose
      dfdp = objf_alpha1
    end  

    gradobjfadj = zeros(D,1);
    t = T
    dt = -dt
 
    # terminal conditions for the adjoint state
    if weight == 1
    	lambdar = zeromat
    	lambdai = zeromat
    elseif weight == 2
      scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
      sr0 = real(scomplex0)
      si0 = imag(scomplex0)

      lambdar = real(conj(scomplex0)*(vtargetr + im*vtargeti))/N 
      lambdai = -imag(conj(scomplex0)*(vtargetr + im*vtargeti))/N
    end
    
    #Backward time stepping loop
    for step in nsteps-1:-1:0
      scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
      sr0 = real(scomplex0)
      si0 = imag(scomplex0)

      if weight == 1
      	hr0 = -weightf1(t,T)/N*(sr0*vtargetr + si0*vtargeti)
      	hi0 =  weightf1(t,T)/N*(sr0*vtargeti - si0*vtargetr)
      end
      
      if penaltyweight == 1
        hr0[N+1:N+Nguard,:] = xi*penalf1(t,T)*vr[N+1:N+Nguard,:]
        hi0[N+1:N+Nguard,:] = xi*penalf1(t,T)*vi[N+1:N+Nguard,:]
      elseif penaltyweight == 2
        hr0 = tinv.*penalf2adj(vr, wmat)
        hi0 = tinv.*penalf2adj(vi, wmat)
      end
     
      # forcing for evolving W (d psi/d alpha1) in the rotating frame
      rotmatrices!(t,domega,rr,ri)
      mul!(dar,rr,amat)
      mul!(dai,ri,amat)
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
          
        # update K and S\         
        KS!(K0, S0, t, amat, adag, domega, splineparams, H0, tmp1, tmp2, tmp3, rr,ri)
        rotmatrices!(t + 0.5*dt*gamma[q],domega,rr,ri)
        KS!(K05, S05, t + 0.5*dt*gamma[q], amat, adag, domega, splineparams, H0, tmp1, tmp2, tmp3,rr,ri)
        rotmatrices!(t + dt*gamma[q],domega,rr,ri)
        KS!(K1, S1, t + dt*gamma[q], amat, adag, domega, splineparams, H0, tmp1, tmp2, tmp3, rr,ri)
          
        # evolve vr, vi
        @inbounds t, vr, vi = timestep.step(t, vr, vi, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident)

        scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti, labframe, t, omega)
        sr1 = real(scomplex1)
        si1 = imag(scomplex1)

        if weight == 1
        	hr1 = -weightf1(t,T)/N*(sr1*vtargetr + si1*vtargeti)
        	hi1 =  weightf1(t,T)/N*(sr1*vtargeti - si1*vtargetr)
        end  
        
        if penaltyweight == 1
          hr1[N+1:N+Nguard,:] = xi*penalf1(t,T)*vr[N+1:N+Nguard,:]
          hi1[N+1:N+Nguard,:] = xi*penalf1(t,T)*vi[N+1:N+Nguard,:]
        else
# needs modification if combined with the weight==1 case above
          hr1 = tinv.*penalf2adj(vr, wmat)
          hi1 = tinv.*penalf2adj(vi, wmat)
        end  
        # evolve lambdar, lambdai
        @inbounds temp, lambdar, lambdai = timestep.step(t0, lambdar, lambdai, dt*gamma[q], hi0, 0.5*(hr0 + hr1), hi1, K0, S0, K05, S05, K1, S1, Ident)

        mul!(dar,rr,amat)
        mul!(dai,ri,amat)
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
      end #for stages
    end # for step (backward time stepping loop)
 
   ineqpengrad = evalineqgrad(pcof, par0, par1)
  
    for k in 1:D
      gradobjfadj[k] = gradobjfadj[k] + ineqpengrad[k]
    end
     
  end # if retadjoint

  if verbose
    println("Inequality penalty: ", ineqpenalty)
    println("Objective functional objfv: ", objfv)
		
    dfdp = dfdp + ineqpengrad[kpar]
    println("Forward integration of gradient of objective function = ", dfdp, " ineqpengrad = ", ineqpengrad[kpar])

    nlast = 1 + nsteps
    println(" Column   Vnrm")
    for q in 1:N
      Vnrm = usaver[:,q,nlast]' * usaver[:,q,nlast] + usavei[:,q,nlast]' * usavei[:,q,nlast]
      Vnrm = sqrt(Vnrm)
      println(q, " | ", Vnrm)
    end

# output final unitary
    println("Final unitary:");
    for q in 1:N
      for c in 1:N
        @printf("%11.4e %+11.4eim ", ufinalr[q,c], ufinali[q,c])
      end
      @printf("\n")
    end

# output primary objective function (infidelity at final time)
    fidelityrot = abs(tracefidcomplex(vfinalr, vfinali, vtargetr, vtargeti, labframe, t, omega))
    println("Final trace fidelity = ", fidelityrot);
    
# Also output L2 norm of last energy level
    normlastguard = zeros(N)
    for q in 1:N
      normlastguard[q] = sqrt( (usaver[Ntot,q,:]' * usaver[Ntot,q,:] + usavei[Ntot,q,:]' * usavei[Ntot,q,:])/nsteps );
    end
    # print the results
    for q in 1:N
      println("Initial data # ", q, " L2 norm last guard level = ", normlastguard[q]) 
    end

    # make plots
    plt1 = plotunitary(usaver + 1im*usavei,T)

    # Evaluate control function at the discrete time levels
    nplot = round(Int64, T*samplerate)
    td = collect(range(1.0/samplerate, stop = T, length = nplot))
    #    @show(td[1])
    #    @show(nplot)

    rplot(t) = rfunc(t,splineparams)
    labdrive = rplot.(collect(td))
    eplotr = cos.(domega[1]*collect(td)) .* labdrive
    eploti = sin.(domega[1]*collect(td)) .* labdrive

    f1 = plot(td, labdrive, lab="", title = "Control function, Lab frame", xlabel = "Time [ns]", ylabel="Amplitude", linewidth = 2)
    f2 = plot(td, eplotr, lab="", title = "Rotating frame, Real part", linewidth = 2)
    f3 = plot(td, eploti, lab="", title = "Rotating frame, Imag part", linewidth = 2)
    # if weight == 1
    #   plot!(td, weightf1.(td,T), lab = "Gate", linewidth = 2)
    # end

#    plt2 = plot(f1, f2, f3, layout = (3,1))        
    plt2 = plot(f1)        

  end #if verbose

# return to calling routine
if verbose && retadjoint
   return objfv, gradobjfadj, plt1, plt2, td, labdrive
elseif verbose
  return objfv, plt1, plt2
elseif retadjoint
  return objfv, gradobjfadj
else
  return objfv
end #if
end

# returns wmat (coefficients for penalty of forbidden states)
function wmatsetup(Ntot::Int64, Ng::Int64)
  w = zeros(Ntot)
  coeff = 2.0
  if Ng > 0
    fact = 0.1
    for q in 0:Ng-1
      w[Ntot-q] = fact^q
    end
  end # if
  wmat = coeff * Diagonal(w)
  return wmat
end

# returns omega
function omegafun(N::Int64)
	omega = zeros(N)
  omega[1] = 0
  omega[2] = 4.10336
  if N >= 3
    omega[3] = 7.98692
  end
  if N >= 4
    omega[4] = 11.65068
  end
  if N >= 5
    omega[5] = 15.09464
  end
  if N >= 6
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
@inline function boundcof(pcof::Array{Float64,1}, D::Int64, maxpar::Float64, eps::Float64)
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
@inline function rotframematrices(Ntot::Int64)
  omega = omegafun(Ntot)
  H0 = zeros(Ntot,Ntot)
  amat = Array(Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U))
  adag = Array(transpose(amat))
  domega = zeros(Ntot)
  domega[1:Ntot-1] = omega[2:Ntot] .- omega[1:Ntot-1]

  return H0, amat, adag, omega, domega
end


@inline function weightf1(t::Float64, T::Float64)
  # period
  tp = T/10
  xi = 4/tp # scale factor
  
  # center time
  tc = T
  tau = (t - tc)/tp
  mask = (tau >= -0.5) & (tau <= 0.5)
  w = xi*64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3
end

@inline function tracefidreal(ur::Array{Float64,2}, vi::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2}, labframe::Bool,t::Float64,omega::Array{Float64,1})
  N = size(vtargetr,2)
  if labframe
    for I in 1:N
  	 rotmatc = cos(omega[I]*t)
  	 rotmats = sin(omega[I]*t)
    
     ua[I] = rotmatc * ur[I] + rotmats * vi[I] # ur = + Re(u), vi = - Im(u)
     va[I] = rotmats * ur[I] - rotmatc * vi[I]
    end
  else
    ua = ur
    va = -vi
  end
 
  fidelity = (tr(ua' * vtargetr + va' * vtargeti)/N)^2 + (tr(ua' * vtargeti - va' * vtargetr)/N)^2
end

@inline function tracefidreal(frcr::Array{Float64,2}, frci::Array{Float64,2}, lambdar::Array{Float64,2}, lambdai::Array{Float64,2})
  fidreal = 0.0
  fidreal = tr(frcr' * lambdar + frci' * lambdai);
end

@inline function tracefidcomplex(ur::Array{Float64,2}, vi::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2}, labframe::Bool, t::Float64, omega::Array{Float64,1})
  N = size(vtargetr,2)
  fidreal = 0.0
  fid_cmplx = tr(ur' * vtargetr .+ vi' * vtargeti)/N + 1im*tr(ur' * vtargeti .- vi' * vtargetr)/N;
end

@inline function  penalf1(t::Float64, T::Float64)
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

function penalf2(vr::Array{Float64,2}, vi::Array{Float64,2},  Nguard::Int64)
  f = 0.0
  if Nguard > 0
    N = size(vr,2)
    rguard = vr[N+1:N+Nguard,:] 
    iguard = vi[N+1:N+Nguard,:]

    for i in 1:Nguard
      w = wf(i,Nguard)
      for j in 1:N
        f = f + w*(rguard[i,j]^2 + iguard[i,j]^2)
      end
    end
 else
    f = 0.0
 end
 return f
end

# anders version
function penalf2a(vr::Array{Float64,N}, vi::Array{Float64,N},  wmat::Diagonal{Float64,Array{Float64,1}}) where N
f = tr( vr' * wmat * vr + vi' * wmat * vi);
return f
end

function penalf2adj(vr::Array{Float64,N}, wmat::Diagonal{Float64,Array{Float64,1}}) where N
f = wmat * vr;
return f
end

function penalf2grad(vr::Array{Float64,N}, vi::Array{Float64,N}, wr::Array{Float64,N}, wi::Array{Float64,N},  wmat::Diagonal{Float64,Array{Float64,1}}) where N
f = tr( wr' * wmat * vr ) + tr( wi' * wmat * vi);
return f
end

function penalf2(v::Array{Float64,2}, Nguard::Int64)
  N = size(v, 2)
  f = zeros(Nguard, N)
  if Nguard > 0
    guard = v[N+1:N+Nguard,:] 
  
    for i in 1:Nguard
      w = wf(i,Nguard)
      for j in 1:N
        f[i,j] =  w*guard[i,j]
      end
    end

 end
 return f
end

function penalf2(vr::Array{Float64,2}, vi::Array{Float64,2}, wr::Array{Float64,2}, wi::Array{Float64,2}, Nguard::Int64)
  Ntot =size(vr,1)
  N = size(vr,2)

  f = 0.0
  if  Nguard > 0 # should give the last guard much higher weight!
    vrguard = vr[N+1:N+Nguard,:]
    viguard = vi[N+1:N+Nguard,:]
    wrguard = wr[N+1:N+Nguard,:]
    wiguard = wi[N+1:N+Nguard,:]   
     for i in 1:Nguard
      w = wf(i,Nguard)
      for j in 1:N
        f = f + w*(vrguard[i,j]*wrguard[i,j] + viguard[i,j]*wiguard[i,j])
      end
    end
  else
    f = 0.0
  end
  return f
end

function wf(i, Nguard)
   w = sqrt((i-1)/Nguard)
end

@inline function normguard(vr::Array{Float64,2}, vi::Array{Float64,2}, Nguard::Int64)
  Ntot =size(vr,1)
  N = size(vr,2)

  f = 0.0
  if Nguard > 0
    rguard = vr[N+1:N+Nguard,:] 
    iguard = vi[N+1:N+Nguard,:]
    f = sum(sum(rguard.^2, dims = 2)) + sum(sum(iguard.^2, dims = 2)) # Is this really a good substitute for sumsq? is it even right?
  else
    f = 0
  end
end

@inline function evalineqpen(pcof::Array{Float64,1}, par_0::Float64, par_1::Float64)
  D = size(pcof,1)
  N = size(pcof,2)
  scalef = 0.1
# testing
#  scalef = 0
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
      titlestr = string("Response to initial data ", ii)
        h = plot(title = titlestr, size = (1000, 1000))
      for jj in 1:Ntot
        labstr = string("State ", jj)
        if ii == 2
         plot!(t, abs.(us[jj,ii,:]), lab = labstr, linewidth = 4, xlabel = "Time [ns]", ylabel = L"\mathrm{ | \psi_i | }")
        else
         plot!(t, abs.(us[jj,ii,:]), leg = false, linewidth = 4, xlabel = "Time [ns]" , ylabel = L"\mathrm{ | \psi_i | }")
        end
    end
    plotarray[ii] = h
  end
  plt = plot(plotarray..., layout = N)
  return plt
end


@inline function evalineqgrad(pcof::Array{Float64,1}, par0::Float64, par1::Float64)
  D = size(pcof,1)
  N = size(pcof,2)
  scalef = 0.1
# testing
#  scalef = 0
  pengrad = zeros(D,N)
  for k in 1:D
    pengrad[k,:] = scalef*(1.0./(par1 .- pcof[k,:]) .- 1.0./(pcof[k,:] .- par0))/D
  end
 return pengrad
end

function screal(vr::Array{Float64,2}, vi::Array{Float64,2}, wr::Array{Float64,2}, wi::Array{Float64,2}, Nguard::Int64,vrguard::Array{Float64,2})
  Ntot =size(vr,1)
  N = size(vr,2)

  f=0;
  if  Nguard > 0 # should give the last guard much higher weight!
    vrguard = vr[N+1:N+Nguard,:]
    viguard = vi[N+1:N+Nguard,:]
    wrguard = wr[N+1:N+Nguard,:]
    wiguard = wi[N+1:N+Nguard,:]   
    f = tr(vrguard*wrguard') +  tr(viguard*wiguard')
  else
    f = 0
  end

end

function KS!(K::Array{Float64,2},S::Array{Float64,2},t::Float64,amat::Array{Float64,2},adag::Array{Float64,2},
            domega::Array{Float64,1},splineparams::bsplines.splineparams,H0::Array{Float64,2},tmp1::Array{Float64,2},
            tmp2::Array{Float64,2},tmp3::Array{Float64,2},rr::Array{Float64,2},ri::Array{Float64,2})
  rrt = transpose(rr)
  rfeval = rfunc(t,splineparams)
  ifeval = ifunc(t,splineparams)
  tmp4 = 0.0
 
  # (rr*amat)
  mul!(tmp1,rr,amat) 
  mul!(tmp2,adag,rrt)
  for  I in eachindex(tmp1)
    # (rr*amat + adag*rr')
   @inbounds tmp3[I] = tmp1[I] + tmp2[I]
  end
   #ri*amat 
  mul!(tmp1,ri,amat)
   # adag*ri
  mul!(tmp2,adag,ri)
  for  I in eachindex(tmp1)
   # (ri*amat + adag*ri)
    @inbounds   tmp4 = tmp1[I] + tmp2[I]
    @inbounds  K[I] = H0[I] + rfeval*tmp3[I] - ifeval*tmp4
  end
  
  # S
  mul!(tmp1,rr,amat)
  mul!(tmp2,adag,transpose(rr))
     
  for  I in eachindex(tmp1)
   @inbounds tmp3[I] = tmp1[I] - tmp2[I]
  end
  
  mul!(tmp1,ri,amat)
  mul!(tmp2,adag,transpose(ri))
   
  for  I in eachindex(tmp1)
   @inbounds tmp4 = tmp1[I] - tmp2[I]  
   @inbounds  S[I]  = ifunc(t,splineparams)*tmp3[I] + rfunc(t,splineparams)*tmp4
  end
end


@inline function rfunc(t::Float64,splineparams::bsplines.splineparams)
 ret = 0.0
 ret = bsplines.bspline2(t,splineparams)
end

@inline function efunc(t::Float64,splineparams::bsplines.splineparams)
  ret = 0.0
  ret = bsplines.bspline2(t,splineparams::bsplines.splineparams)
end

@inline function ifunc(t::Float64,splineparams::bsplines.splineparams)
  ret = 0.0
end

function rotmatrices!(t::Float64, domega::Array{Float64,1},rr::Array{Float64,2},ri::Array{Float64,2})
 for I in 1:length(domega)
  rr[I,I] = cos(domega[I]*t)
  ri[I,I] = -sin(domega[I]*t)
 end
end

function grupdate(gr0::Array{Float64,2}, gi0::Array{Float64,2}, dai::Array{Float64,2}, dar::Array{Float64,2}, vr::Array{Float64,2}, vi::Array{Float64,2},rfalpha::Float64,ifalpha::Float64)
   gr0 = rfalpha.*( (dai .-  dai')*vr .- (dar .+ dar')*vi) .+ ifalpha.*(  (dai .+ dai')*vi .+ (dar .-  dar')*vr) #should it really be ifalpha' and ralpha' here?? Different n anders code ..
   gi0 = rfalpha.*( (dar .+  dar')*vr .+ (dai .- dai')*vi) .+ ifalpha.*( -(dai .+ dai')*vr .+ (dar .-  dar')*vi)
   return gr0, gi0
end
end
