# bsplinetest(Nspline)
#
# INPUT:
# Nspline: number of splines >=3

function  bsplinetest(Nspline::Int64,test = false)
	if Nspline < 3
		error("Input has to be larger than 3") # TODO: Should be possible to use something like "assert" instead of if
	end
	
	T = 120.0#Final time
	dt  = 1/20
	if test
		dt = 40
    end
	Nsteps = ceil(Int64,T/dt)
	dt = T/Nsteps

	println("Final time = ", T, ", number of steps = ", Nsteps, ", time step =", dt)
    
    # TODO: write this part as a constructor to param
	Nknots = Nspline + 1
	dtknot = T/(Nspline - 2)
	width = 3*dtknot
	tcenter = dtknot*(collect(1:Nspline) .- 1.5) # TODO: check if it is efficient to use collect
	tknot = dtknot*(collect(1:Nknots) .- 2)

	# Generate random cofs
	#pcof = 2*rand(Nspline,1) .- 1 

	#s Anders spline cofs
	#pcof = zeros(Nspline)
	#i1 = round(Int64,0.5*Nspline)
	#pcof[i1] = 1
	#pcof[i1+1] = -1

	# osccilating pcof
	pcof = ones(Nspline);

	for i  in 1:2:Nspline
		pcof[i] = -1 
	end


	param = bsplines.splineparams(T, Nspline,Nknots,tcenter,tknot,dtknot,pcof)
	@show(typeof(param))
	
	td = collect(range(0,length = Nsteps, stop = T )) #TODO: Should we colect this or not?

	ctrl = bsplines.bspline2(td,param)

	# Attempt to plot frequiencies that does not work

	#df = 1/T;
  	#Nf = Nsteps;
  	#freq=[ -(ceil.(Int64, (Nf-1)/2):-1:1); 0; (1:floor((Nf-1)/2)) ] * df; # In Hz
	#om = 2*pi*freq; # angular frequency in rad/sec

  	#fctrl = fftshift(fft(ifftshift(ctrl)));
  	#fctrl = fctrl./Nf;

  	#plot(om,abs.(fctrl).+1e-18, yaxis=:log,line=(:dot, 4))
    
    t = 12.34
	g = bsplines.gradbspline2(t,param)

	k = max.(3, ceil.(Int64,t./dtknot .+ 2)) 
    dp = 1e-5;
	f0 = bsplines.bspline2(t,param)
	q = k.-2
	
	param.pcof[q] = param.pcof[q] .+ dp
	f1 = bsplines.bspline2(t,param)

	println("g(", t, ") =", g[q], " and  (f1 - f0)/dp = ", (f1-f0)/dp)
	plot(td, ctrl, legend = false, xaxis = "Time [s]")

	if test
		return ctrl, g
	end

end