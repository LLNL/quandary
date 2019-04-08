
using Optim
	N = 2
	
	Nguard = 3

	Ntot = N + Nguard
	
        utarget = zeros(ComplexF64,Ntot,N)

	# pi/2 y-rot gate
        utarget[1,1] = 1/sqrt(2)
        utarget[1,2] = -1/sqrt(2)
        utarget[2,1] = 1/sqrt(2)
        utarget[2,2] = 1/sqrt(2)

	# pi/2 x-rot gate

        # utarget[1,1] = 1.0/sqrt(2)
        # utarget[1,2] = -1im*1.0/sqrt(2)
        # utarget[2,1] = -1im*1.0/sqrt(2)
        # utarget[2,2] = 1.0/sqrt(2)

	cfl = 0.05

	T = 100.0

	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
	

#	pcof0  = zeros(351) 
	pcof0  = zeros(200) 
	#pcof0 = (rand(250) .- 0.5).*maxpar*0.1
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

   objv, grad, pl1, pl2 = objfunc.traceobjgrad(pcof,params,order, true, true, weight, penaltyweight)

   println("Objfunc = ", objv)
   pl1


	
  	 
	
