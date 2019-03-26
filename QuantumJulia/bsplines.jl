module bsplines

struct splineparams #todo specify tyes?
	T
    Nspline
    Nknots
    tcenter
    tknot
    dtknot
    pcof
end	

#TODO: Add a constructor here??


# bspline2: Evaluate quadratic bspline function
function bspline2(t,param::splineparams)
	f = zeros(length(t))

	dtknot = param.dtknot
	tknot = param.tknot
	width = 3*dtknot

	k = max.(3, ceil.(Int64,t./dtknot .+ 2)) # Unsure if this line does what it is supposed to
	k = min.(k, param.Nspline)

	# 1st segment of nurb k
	tc = param.tcenter[k]
	tau = (t .- tc)./width
	f = f .+ param.pcof[k] .* (9/8 .+ 4.5.*tau .+ 4.5.*tau.^2) # test to remove square for extra speed

	# 2nd segment of nurb k-1
	tc = param.tcenter[k.-1]
	tau = (t .- tc)./width
	f = f .+ param.pcof[k.-1] .* (0.75 .- 9 .*tau.^2)

	# 3rd segment of nurb k-2
	tc = param.tcenter[k.-2]
	tau = (t .- tc)./width
	f = f .+ param.pcof[k.-2] .* (9/8 .- 4.5.*tau .+ 4.5.*tau.^2)
end

# gradbspline2: Evaluate gradient with respect to pcof of quadratic bspline
function gradbspline2(t,param::splineparams)
	g = zeros(param.Nspline)
	dtknot = param.dtknot
	tknot = param.tknot
	width = 3*dtknot 

	k = max.(3, ceil.(Int64,t./dtknot .+ 2)) # t_knot(k-1) < t <= t_knot(k), but t=0 needs to give k=3
	k = min.(k, param.Nspline) # protect agains roundoff that sometimes makes t/dt > N_nurbs-2

     #1st segment of nurb k
    tc = param.tcenter[k]
	tau = (t .- tc)./width
	g[k] = (9/8 .+ 4.5.*tau .+ 4.5.*tau.^2);

	#2nd segment of nurb k-1
	tc = param.tcenter[k.-1]
	tau = (t .- tc)./width
	g[k.-1] = (0.75 .- 9 .*tau.^2) #g + g=... ?

	# 3rd segment og nurb k-1
    tc = param.tcenter[k.-2]
	tau = (t .- tc)./width
	g[k.-2] = (9/8 .- 4.5.*tau .+ 4.5.*tau.^2);
	return g
end

end