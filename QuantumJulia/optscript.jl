
using Optim
	N = 4
	
	Nguard = 3
	Ntot = N + Nguard
	
	Ident = Matrix{Float64}(I, Ntot, Ntot)   
	utarget = Ident[1:Ntot,1:N]
	utarget[:,3] = Ident[:,4]
	utarget[:,4] = Ident[:,3]

	cfl = 0.05

	T = 150.0

	testadjoint = 0
	maxpar =0.09
	
	params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
	

	pcof0  = zeros(200)
	pcof0[5] = 0.05
    pcof0[6] = -0.05
	order = 2

    function f(pcof)
    #@show(pcof)
     f =objfunc.traceobjgrad(pcof,params,order,false,false)
    # @show(f)
     return f[1]
     end

    function g!(G,pcof,params,order)
    	objf, Gtemp = objfunc.traceobjgrad(pcof,params,order,false, true)
    	
    	Gtemp = vcat(Gtemp...) 
    	for i in 1:length(Gtemp)
    	  G[i] = Gtemp[i]
    	end
    end

   gopt!(G,pcof) = g!(G,pcof,params,order)

   pcof = Optim.minimizer(optimize(f, gopt!, pcof0, BFGS()))

   pl1, pl2, objv, grad = objfunc.traceobjgrad(pcof,params,order, true,true)

   println("Objfunc = ", objv)
   pl1


	
  	 
	
