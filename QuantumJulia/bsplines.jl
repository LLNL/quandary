module bsplines

struct splineparams #todo specify types?
  T::Float64
  Nspline::Int64
  Nknots::Int64
  tcenter::Array{Float64,1}
  tknot::Array{Float64,1}
  dtknot::Float64
  pcof::Array{Float64,1}
end	

#TODO: Add a constructor here??


# bspline2: Evaluate quadratic bspline function

@inline function bspline2(t::Float64, param::splineparams)
  	f = 0.0

	dtknot = param.dtknot
	tknot = param.tknot
	width = 3*dtknot

	k = max.(3, ceil.(Int64,t./dtknot + 2)) # Unsure if this line does what it is supposed to
	k = min.(k, param.Nspline)

	# 1st segment of nurb k
	tc = param.tcenter[k]
	tau = (t .- tc)./width
	f = f + param.pcof[k] * (9/8 .+ 4.5*tau + 4.5*tau^2) # test to remove square for extra speed

	# 2nd segment of nurb k-1
	tc = param.tcenter[k-1]
	tau = (t - tc)./width
	f = f .+ param.pcof[k.-1] .* (0.75 - 9 *tau^2)

	# 3rd segment of nurb k-2
	tc = param.tcenter[k.-2]
	tau = (t .- tc)./width
	f = f + param.pcof[k.-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
end

# real part
@inline function bspline2r(t::Float64, param::splineparams)
  	f = 0.0

	dtknot = param.dtknot
	tknot = param.tknot
	width = 3*dtknot

	k = max.(3, ceil.(Int64,t./dtknot + 2)) # Unsure if this line does what it is supposed to
	k = min.(k, param.Nspline)

	# 1st segment of nurb k
	tc = param.tcenter[k]
	tau = (t .- tc)./width
	f = f + param.pcof[k] * (9/8 .+ 4.5*tau + 4.5*tau^2) # test to remove square for extra speed

	# 2nd segment of nurb k-1
	tc = param.tcenter[k-1]
	tau = (t - tc)./width
	f = f .+ param.pcof[k.-1] .* (0.75 - 9 *tau^2)

	# 3rd segment of nurb k-2
	tc = param.tcenter[k.-2]
	tau = (t .- tc)./width
	f = f + param.pcof[k.-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
end

# imaginary part
@inline function bspline2i(t::Float64, param::splineparams)
  	f = 0.0

	dtknot = param.dtknot
	tknot = param.tknot
	width = 3*dtknot
        D = param.Nspline # offset for the imaginary coefficients

	k = max.(3, ceil.(Int64,t./dtknot + 2)) # Unsure if this line does what it is supposed to
	k = min.(k, param.Nspline)

	# 1st segment of nurb k
	tc = param.tcenter[k]
	tau = (t .- tc)./width
	f = f + param.pcof[D+k] * (9/8 .+ 4.5*tau + 4.5*tau^2) # test to remove square for extra speed

	# 2nd segment of nurb k-1
	tc = param.tcenter[k-1]
	tau = (t - tc)./width
	f = f .+ param.pcof[D+k.-1] .* (0.75 - 9 *tau^2)

	# 3rd segment of nurb k-2
	tc = param.tcenter[k.-2]
	tau = (t .- tc)./width
	f = f + param.pcof[D+k.-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
end

@inline function bspline2(t::Array{Float64,1},param::splineparams)
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

# gradbspline2r: Real part. gradient with respect to pcof of quadratic bspline
@inline function gradbspline2r(t::Float64,param::splineparams)
  g = zeros(2*param.Nspline) # real and imag parts
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

  # 3rd segment og nurb k-2
  tc = param.tcenter[k.-2]
  tau = (t .- tc)./width
  g[k.-2] = (9/8 .- 4.5.*tau .+ 4.5.*tau.^2);
  return g
end

# gradbspline2i: Imaginary part. gradient with respect to pcof of quadratic bspline (identical to the real part)
@inline function gradbspline2i(t::Float64,param::splineparams)
  g = zeros(2*param.Nspline) # real and imag parts
  dtknot = param.dtknot
  tknot = param.tknot
  width = 3*dtknot 
  D = param.Nspline # offset for the imaginary coefficients

  k = max.(3, ceil.(Int64,t./dtknot .+ 2)) # t_knot(k-1) < t <= t_knot(k), but t=0 needs to give k=3
  k = min.(k, param.Nspline) # protect agains roundoff that sometimes makes t/dt > N_nurbs-2

  #1st segment of nurb k
  tc = param.tcenter[k]
  tau = (t .- tc)./width
  g[k .+ D] = (9/8 .+ 4.5.*tau .+ 4.5.*tau.^2);

  #2nd segment of nurb k-1
  tc = param.tcenter[k.-1]
  tau = (t .- tc)./width
  g[k.-1 .+ D] = (0.75 .- 9 .*tau.^2) 

  # 3rd segment og nurb k-2
  tc = param.tcenter[k.-2]
  tau = (t .- tc)./width
  g[k.-2 .+ D] = (9/8 .- 4.5.*tau .+ 4.5.*tau.^2);
  return g
end

# gradbspline2: Evaluate gradient with respect to pcof of quadratic bspline
@inline function gradbspline2(t::Float64,param::splineparams)
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
