module timestep

struct stormerverlet 
	K::Function
	S::Function
	In::Array{Float64,2}
end

# Force function
@inline function step(stepper::stormerverlet, t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64, uforce::Function, vforce::Function) where N
	vforce0 = vforce(t)
	#uforce05 = uforce(t + 0.5*h)
	uforce05 = 0.5*(uforce(t) + uforce(t + h))

	vforce1 = vforce(t + h)

    t,u,v = step(stepper, t, u, v, h, vforce0, uforce05, vforce1)
	
	return t, u, v
end

# wWthout forcing
@inline function step(stepper::stormerverlet, t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64) where N
	Ntot = size(u)
	vforce0 = zeros(Ntot) 
	uforce05 = zeros(Ntot)
	vforce1 = zeros(Ntot)


    t,u,v = step(stepper, t, u, v, h, vforce0, uforce05, vforce1)
	
	return t, u, v
end

# Forcing as array 
@inline function step(stepper::stormerverlet, t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64, vforce0::Array{Float64,N}, uforce05::Array{Float64,N}, vforce1::Array{Float64,N}) where N
	S = stepper.S
	K = stepper.K
	In = stepper.In

	Kt  = K(t)
	St  = S(t)

	Kt05h = K(t +0.5*h)
	St05h = S(t + 0.5*h)

    rhs = (Kt*u .+  St*v .+ vforce0)
	l1 = (In .-  0.5*h.*St)\rhs
	kappa1 = St05h*u .- Kt05h*(v .+ 0.5*h.*l1) .+ uforce05
	v05 = (v .+ 0.5*h.*l1)
	rhs = St05h*(u .+ 0.5*h*kappa1) .- Kt05h*v05 .+ uforce05
	kappa2 = (In .- (0.5*h).*St05h)\rhs

	u = u .+ (0.5*h).*(kappa1 .+ kappa2)
	l2 = K(t +h)*u .+ S(t + h)*v05 .+ vforce1

	v = v .+ 0.5*h.*(l1 .+ l2)
	t = t + h

	return t, u, v
end

@inline function step(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64,
 vforce0::Array{Float64,N}, uforce05::Array{Float64,N}, vforce1::Array{Float64,N},
  K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N}, K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N}) where N
    rhs = (K0*u .+  S0*v .+ vforce0)
	l1 = (In .-  0.5*h.*S0)\rhs
	kappa1 = S05*u .- K05*(v .+ 0.5*h.*l1) .+ uforce05
	v05 = (v .+ 0.5*h.*l1)
	rhs = S05*(u .+ 0.5*h*kappa1) .- K05*v05 .+ uforce05
	kappa2 = (In .- (0.5*h).*S05)\rhs

	u = u .+ (0.5*h).*(kappa1 .+ kappa2)
	l2 = K1*u .+ S1*v05 .+ vforce1

	v = v .+ 0.5*h.*(l1 .+ l2)
	t = t + h

	return t, u, v
end

@inline function step(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64,
  K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N}, K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N}) where N
    rhs = (K0*u .+  S0*v)
	l1 = (In .-  0.5*h.*S0)\rhs
	kappa1 = S05*u .- K05*(v .+ 0.5*h.*l1) 
	v05 = (v .+ 0.5*h.*l1)
	rhs = S05*(u .+ 0.5*h*kappa1) .- K05*v05 
	kappa2 = (In .- (0.5*h).*S05)\rhs

	u = u .+ (0.5*h).*(kappa1 .+ kappa2)
	l2 = K1*u .+ S1*v05 

	v = v .+ 0.5*h.*(l1 .+ l2)
	t = t + h
	return t, u, v
end

function stepseparable(stepper::stormerverlet,u,v,t,h)
  if S(stepper.t) != 0
  	error("S has to be zero for separble time-stepping to ework")
  end

	S = stepper.S
	K = stepper.K
	In = stepper.In
	h = stepper.h
	uforce = stepper.uforce
	vforce = stepper.vforce
    
	l1 = (K(t)*u .+ vforce(t))
	kappa1 = - K(t + 0.5*h)*(v .+ 0.5*h.*l1) .+ uforce(t + 0.5*h)
	kappa2 = kappa1

	u = u .+ (h/2).*(kappa1 .+ kappa2)
	l2 = K(t +h)*u  .+ vforce(t + h)

	v = v .+ 0.5*h.*(l1 .+ l2)
	t = t + h

	return t, u, v
end

function getgamma(order,stages = [])
 if order == 2	# 2nd order basic verlet
	stages = 1
    gamma = [1]
  elseif order == 4 # 4th order Composition of Stromer-Verlet methods
    stages=3
    gamma = zeros(stages)
    gamma[1] = gamma[3] = 1/(2 - 2^(1/3))
    gamma[2] = -2^(1/3)*gamma[1]
  elseif order == 6 # Yoshida (1990) 6th order, 7 stage method
    if stages==7
      gamma = zeros(stages)
      gamma[2] = gamma[6] = 0.23557321335935813368479318
      gamma[1] = gamma[7] = 0.78451361047755726381949763
      gamma[3] = gamma[5] = -1.17767998417887100694641568
      gamma[4] = 1.31518632068391121888424973
    else # Kahan + Li 6th order, 9 stage method
      stages=9;
      gamma = zeros(stages)
      gamma[1]= gamma[9]= 0.39216144400731413927925056
      gamma[2]= gamma[8]= 0.33259913678935943859974864
      gamma[3]= gamma[7]= -0.70624617255763935980996482
      gamma[4]= gamma[6]= 0.08221359629355080023149045
      gamma[5]= 0.79854399093482996339895035
    end
   end
   
   return gamma, stages
end
end