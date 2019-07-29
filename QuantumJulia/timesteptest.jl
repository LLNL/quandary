using LinearAlgebra
using Plots
include("timestep.jl")

function timesteptest( cfl = 0.1, testcase = 1, order = 2, plotcomp = "err",test = false)

	N = 2 # vector dimension	
	IN = Matrix{Float64}(I, N, N)
	
	if testcase == 1 || testcase == 2
		K0 = [0 1; 1 0]
		S0 = [0 0; 0 0]
	elseif testcase == 0 || testcase == 3
	    K0 = [0 0; 0 0]
	    S0 = [0 1; -1 0]
	end
	
	#Final time
	period = 1
	T = 5pi
	omega = 2*pi/period
	
	println("Testcase: ", testcase, "  Cfl: ", cfl, " Final time: ", T )
	
	lamb = eigvals(K0 .+ S0)
	maxeig = maximum(abs.(lamb))
	
	#time step
	dt = cfl/maxeig  
	nsteps = ceil(Int64,T/dt) 
	dt = T/nsteps
	
	# Initial conditions
	u = [1.0; 0.0]
	v = [0.0; 0.0]
	t = 0.0
	
	# timefunctions and forcing	
	if testcase == 1
		timefunc1(t) = 0.5*(sin(0.5*omega*(t)))^2;
		uforce1(t::Float64) = [0.0; 0.0]
		vforce1(t::Float64) = [0.0; 0.0]

		timefunc =timefunc1
		uforce = uforce1
		vforce = vforce1

	elseif testcase == 0
		timefunc0(t::Float64) = 0.25*(1-sin(omega*t))
		uforce0(t::Float64) = [0.0; 0.0]
		vforce0(t::Float64) = [0.0; 0.0]

		timefunc =timefunc0
		uforce = uforce0
		vforce = vforce0
	elseif testcase == 2
		timefunc2(t::Float64) = 4/T^2 *t*(T-t)
		phi12(t::Float64) = 0.25*(t - sin(omega*t)/omega)
		phidot2(t::Float64) = 0.5*(sin(0.5*omega*(t)))^2
		uforce2(t::Float64) = [(timefunc2(t) - phidot2(t))*sin(phi12(t)); 0.0]
		vforce2(t::Float64) = [0.0; -(timefunc2(t) - phidot2(t)) * cos(phi12(t))]

		timefunc =timefunc2
		uforce = uforce2
		vforce = vforce2
	elseif testcase == 3
		timefunc3(t::Float64) =  4/T^2 *t*(T-t)
		phi13(t::Float64) = 0.25*(t - sin(omega*t)/omega)
		phidot3(t::Float64) = 0.5*(sin(0.5*omega*(t)))^2
		uforce3(t::Float64) = [-phidot3(t)*sin(phi13(t)); timefunc(t)*cos(phi13(t))]
	 	vforce3(t::Float64) = [-timefunc(t)*sin(phi13(t)); phidot3(t)*cos(phi13(t))]

	 	timefunc =timefunc3
		uforce = uforce3
		vforce = vforce3
	end	
	
	K(t::Float64) = timefunc(t)*K0
	S(t::Float64) = timefunc(t)*S0


	#Create time stepper
	gamma, stages = timestep.getgamma(order)
	timestepper = timestep.stormerverlet(K,S,IN)
    
	#Time integration
    usave = u
	vsave = -v
	tsave = t
	start = time()
	#nsteps = 1
	for ii in 1:nsteps
	   for jj in 1:stages 
	 @inbounds 	 t, u, v =  timestep.step(timestepper,t,u,v,dt*gamma[jj],uforce,vforce)
	   end
	   usave = [usave u]
	   vsave = [vsave -v]
	   tsave = [tsave t]
	end
	elapsed = time() - start
	@show(elapsed)

	#Compute anal
	if testcase == 1 || testcase == 2 || testcase==3
      phi = 0.25.*(tsave - 1/omega.*sin.(omega.*tsave))
      cg = cos.(phi)
      ce = -1im*sin.(phi)
    elseif testcase == 0
      phi = 0.25.*( tsave + 1/omega.*(cos.(omega.*tsave) .- 1) );
      cg = cos.(phi);
      ce = -sin.(phi);
    end

    #compute errors
    cg_err = sqrt((usave[1,end]-real(cg[end]))^2 + (vsave[1,end]-imag(cg[end]))^2);
    ce_err = sqrt((usave[2,end]-real(ce[end]))^2 + (vsave[2,end]-imag(ce[end]))^2);

    println("cg-err = " , cg_err, " ce-err = ", ce_err)


    if plotcomp == "err"
    	titlestr = string("Err, test ", testcase)
    	ple = plot(tsave',usave[1,:] .- real(cg)',label = "Re(err-1)", title = titlestr)
    	plot!(tsave',vsave[1,:] .- imag.(cg)',label = "Im(err-1)")
    	plot!(tsave',usave[2,:].- real.(ce)',label = "Re(err-2)")	
    	plot!(tsave',vsave[2,:] .- imag.(ce)',label = "Im(err-2)" )
        display(ple)
    elseif plotcomp == "sol"
    	titlestr = string("Sol, test ", testcase)
    	pls = plot(tsave',usave[1,:] ,label = "Re(1)", title = titlestr)
   	    plot!(tsave',vsave[1,:],label = "Im(1)")
    	plot!(tsave',usave[2,:] ,label = "Re(2)")	
    	plot!(tsave',vsave[2,:] ,label = "Im(2)" )
        display(pls)
    end
if test
	return t,u,v
end
	
end
