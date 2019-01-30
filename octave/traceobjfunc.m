%-*-octave-*--
%
% traceobjfunc: solve a model problem from quantum control theory
%
% USAGE:
% 
% [cost, ufinal] = traceobjfunc(a1, verbose)
%
% INPUT:
% a1(D,1): amplitudes of the control functions as a D x 1 column vector, D=size(a1,1)
%
% OUTPUT:
% cost: trace norm of gate infidelity cost functional
% ufinal: state vector at t=T
%
function [cost uFinal] = traceobjfunc(a1, verbose)

  order = 6;
  if (order == 6)
    stages = 9;
  end
  abs_or_real=1; # plot the magnitude (abs) of real part of the solution (1 for real)

  if nargin < 1
    a1 = 1.0;
  end

  if nargin < 2
    verbose=0;
  end

  cfl = 0.1;

  N = 4; # vector dimension

  D = size(a1,1); # parameter dimension

  ## if (mod(D,2) == 1)
  ##   printf("ERROR: D=%d, is ODD\n", D);
  ##   return;
  ## end
  
# coefficients in H0
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

  H0 = diag([d0, d1, d2, d3]);

  K1 = [0, 1, 0, 0;
  	1, 0, sqrt(2), 0;
  	0, sqrt(2), 0, sqrt(3);
  	0, 0, sqrt(3), 0];

  S1 = [0, 1, 0, 0;
  	-1, 0, sqrt(2), 0;
  	0, -sqrt(2), 0, sqrt(3);
  	0, 0, -sqrt(3), 0];

#  S1 = zeros(N,N);
# final time
  T = 15;

  if (verbose)
    printf("Vector dim (N) = %d, Param dim (D) = %d, a1(1) = %e, Final time = %e, CFL = %e\n", N, D, a1(1), T, cfl);
  end

		     # first evaluate the polynomials on a coarse grid
  pad0 = timefunc(D, 100);
  ptot0 = pad0*a1;
  [pmax imax] = max(ptot0); # assumes ptot is real-valued
  [pmin imin] = min(ptot0);

# estimate largest eigenvalue
  t = (imax-1)/100 * T;

  H=H0+pmax*K1;
  lambda = eig(H);
  maxeig1 = norm(lambda,"inf");
  if (verbose)
    printf("(t, pmax, maxeig) = (%e, %e, %e)\n", t, pmax, maxeig1);
  end

  t = (imin-1)/100 * T;

  H=H0+pmin*K1;
  lambda = eig(H);
  maxeig2 = norm(lambda,"inf");
  if (verbose)
    printf("(t, pmin, maxeig) = (%e, %e, %e)\n", t, pmin, maxeig2);
  end

# tmp
#  return

  maxeig = max(maxeig1, maxeig2);
  
  cfunc = zeros(N,1);
  beta = zeros(N,1);
  cu_sp = zeros(N,1);
  ca1 = zeros(N,1);
  vect = zeros(N,1);

		   # Final time T
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*K1 estimated by maxeig
  nsteps = ceil(T/dt);
  dt = T/nsteps;
  if (verbose)
    printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   T, nsteps, maxeig, cfl, dt);
  end

# evaluate the polynomials at the discrete time levels
# evaluate all polynomials on the midpoint grid
  [pad, td] = timefunc(D, nsteps);
  qad = pad; # tmp
  b1 = 0.*a1;
# sum up all polynomial components
  ptot = pad*a1;
  qtot = qad*b1; # tmp

# form the weight function
  tp = 0.125*T;
  t0 = T;
  tau = (td - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);
  wghf1 = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;

# different weight functions for different components
  wconst = 0.01; # for response to e2 and e3
  
# the basis for the initial data as a matrix
  Ident=diag([1, 1, 1, 1]);
  U0 = Ident;

# Target state at t=T
  uTarget = [0, 1, 0, 0;
	     1, 0, 0, 0;
	     0, 0, 1, 0;
	     0, 0, 0, 1];

# initial data and allocation of solution vectors
  uSol = U0;
# real and negative imaginary part of the solution
  ur = U0;
  vi = zeros(N, N);

# RK stage variables
  kay1 = zeros(N,N);
  kay2 = zeros(N,N);
  ell1 = zeros(N,N);
  ell2 = zeros(N,N);

# zero forcing
  fv_0 = zeros(N,N);
  fv_1 = zeros(N,N);
  fu_1o2 = zeros(N,N);

  if (order == 2)	# 2nd order basic verlet
    stages = 1;
    gamma(1) = 1;
  elseif (order == 4) # 4th order Composition of Stromer-Verlet methods
    order = 4;
    stages=3;
    gamma = zeros(stages,1);
    gamma(1) = gamma(3) = 1/(2 - 2^(1/3));
    gamma(2) = -2^(1/3)*gamma(1);
  elseif (order == 6) # Yoshida (1990) 6th order, 7 stage method
    if (stages==7)
      gamma = zeros(stages,1);
      gamma(2) = gamma(6) = 0.23557321335935813368479318;
      gamma(1) = gamma(7) = 0.78451361047755726381949763;
      gamma(3) = gamma(5) = -1.17767998417887100694641568;
      gamma(4) = 1.31518632068391121888424973;
    else # Kahan + Li 6th order, 9 stage method
      stages=9;
      gamma = zeros(stages,1);
      gamma(1)= gamma(9)= 0.39216144400731413927925056;
      gamma(2)= gamma(8)= 0.33259913678935943859974864;
      gamma(3)= gamma(7)= -0.70624617255763935980996482;
      gamma(4)= gamma(6)= 0.08221359629355080023149045;
      gamma(5)= 0.79854399093482996339895035;
    end
  end
  
# testing octave syntax
  ## step=1;
  ## S = qtot(step).*S1; # skew-symmetric part
  ## rhs = H*ur + S*vi;
  ## l1 = linsolve( Ident-0.5*dt*S, rhs ); # - 0.5*dt*S);
  ## printf("Size(Ident)= (%d, %d), size(rhs)=(%d,%d), size(l1)=(%d,%d)\n", size(Ident), size(rhs), size(l1));#
# end test

  v1 = zeros(N,1);
  v2 = zeros(N,1);
# for computing the cost function
  cost = 0;

			     # time stepping loop, harmonic oscillator
  if (verbose)
    usave = zeros(N,N,nsteps+1);
    usaver = zeros(N,N,nsteps+1);
    usavei = zeros(N,N,nsteps+1);
    usave(:,:,1) = uSol;
    usaver(:,:,1) = ur;
    usavei(:,:,1) = -vi;
  end

# handles to time and forcing functions
  tfunc = @tf1;
  uforce = @uzero;
  vforce = @vzero;  
  
  separable = (norm(S1) < 1e-15);
  printf("Separable = %d\n", separable);

  t=0;
  tm=0;
  step=0;
  for step=1:nsteps
# 2nd order Magnus integrator
    H = H0 + tf1(tm+0.5*dt, a1).*(K1 + I*S1); # symmetric + skew-symmtric 
    expH = expm(-I*dt*H);

    uSol = expH * uSol;

    for q=1:stages
      [ur, vi, t] = stromer_verlet_mat(ur, vi, tfunc, t, gamma(q)*dt, a1, H0, K1, S1, Ident, separable, uforce, vforce); # t, ur, vr are updated
    end

				# Evaluate time function
    ## tf_0 = tfunc(t, a1);
    ## tf_1o2 = tfunc(t+0.5*dt, a1);
    ## tf_1 = tfunc(t+dt, a1);
    
    ## 		    # Partitioned 2nd order RK method (Stromer-Verlet)
    ## ell1 = (H0+tf_0*K1)*ur + fv_0;
    ## kay1 = - (H0 + tf_1o2*K1)*(vi+0.5*dt*ell1) + fu_1o2;
    ## kay2 = kay1;
    ## ell2 = (H0+tf_1*K1)*(ur+0.5*dt*(kay1+kay2)) + fv_1;

    ## ur = ur + 0.5*dt*(kay1 + kay2);
    ## vi = vi + 0.5*dt*(ell1 + ell2);

    tm = tm+dt; # for Magnus integrator

				# accumulate cost = integral ( 1 - Tr( uSol' * w(t) * uTarget ) )
    fidelity = trace(ctranspose(uSol) * uTarget)/N;
    cost = cost + dt*wghf1(step)*( 1 - abs(fidelity)^2 );

				# evaluate energy
    if (verbose)
      usave(:,:,step+1) = uSol;
      usaver(:,:,step+1) = ur;
      usavei(:,:,step+1) = -vi;
      
#      printf("Time = %e, |fidelity| = %e, weight = %e\n", t, abs(fidelity), wghf1(step));

      ## if (mod(step,1000)==0 || step==nsteps-1)
      ## 	v1 = uTime(:,step+1) + uTime(:,step);
      ## 	v2 = uTime(:,step+1) - uTime(:,step);
      ## 	energy = 0.25*(norm(v1)^2 - norm(v2)^2);
      ## 	energy = uTime(:,step+1)'*H*uTime(:,step+1);
	
      ## 	printf("Time step = %d, time = %e, energy = %e\n", step, t, energy);
      ## end
    end # if verbose
  end # for (time stepping loop)

  uFinal = uSol;

				# plot results
  if (verbose)
				# difference at final time
    Nplot = length(usave(1,1,:));
    printf("CFL=%g, Initial data   Component  abs(real)  abs(imag)\n", cfl);
    for q=1:N
      for c=1:N
	printf("  %16d  %8d  %15.8e %15.8e\n", q, c, abs(usaver(c,q,Nplot) - real(usave(c,q,Nplot))),  abs(usavei(c,q,Nplot) - imag(usave(c,q,Nplot))) );
      end
    end
				# unitary?
    printf(" Initial data  Mnrm   Vnrm\n");
    for q=1:N
      Vnrm = usaver(:,q,Nplot)' * usaver(:,q,Nplot) + usavei(:,q,Nplot)' * usavei(:,q,Nplot);
      Vnrm = sqrt(Vnrm);
      Mnrm = norm(usave(:,q,Nplot));
      printf(" %d  %e  %e\n", q, Mnrm, Vnrm);
    end
			    # tmp: compare solutions from both methods

    tplot = linspace(0,T,Nplot);
    c=3;
    q=3;
				# real part
    figure(1);
    h=plot(tplot, real(usave(c,q,:))- usaver(c,q,:));
    tstr = sprintf("Difference, component %d\n", c);
    title(tstr);
    legend("Re(Magnus - Verlet)", "location", "east");
    axis tight

				# imaginary part
    figure(2);
    h=plot( tplot, imag(usave(c,q,:))- usavei(c,q,:));
    tstr = sprintf("Difference, component %d\n", c);
    title(tstr);
    legend( "Im(Magnus-Verlet)", "location", "east");
    axis tight

#    plotunitary(usaver, T, abs_or_real);
#    plotunitary(usave, T, abs_or_real);
    
   figure(5);
   subplot(2,1,1);
   h=plot(td, ptot);
   axis("tight");
    ## pmin = min(ptot);
    ## pmax = max(ptot);
    ## pdelta=pmax-pmin;
    ## axis([0,T,pmin-0.1*pdelta,pmax+0.1*pdelta]);
   set(h,"linewidth",2);
   title("Forcing function");

    subplot(2,1,2);
    h = plot(td, wghf1, "m");
    axis tight;
    set(h,"linewidth",2);
    title("Weight function");
#    legend("e0 & e1", "e2 & e3","location","north");
    ## h = plot( [1:D/2], a1(1:2:end), "b*",  [1:D/2], a1(2:2:end), "r*");
    ## set(h,"markersize",10);
    ## amin = min(a1);
    ## amax= max(a1);
    ## adelta=amax-amin;
    ## axis([0.5 D/2+0.5, amin-0.1*adelta, amax+0.1*adelta]);
    ## legend("cos", "sin", "location", "east");
    ## title("Parameters");
  end

				# total cost function at final time
  finalFidelity = trace(ctranspose(uFinal) * uTarget)/N;
  finalCost = 1 - abs(finalFidelity);

  if (verbose)
    printf("Forward calculation: Parameter a1 =[ %e", a1(1));
    for q=2:D
      printf(", %e", a1(q));
    end
    printf(" ]\n");
				# check if uFinal is unitary
    utest = ctranspose(uFinal) * uFinal - U0;
    printf("Final unitary infidelity = %e, Final |gate fidelity| = %e, and |trace| gate infidelity = %e\n", norm(utest), abs(finalFidelity), finalCost);
    printf("Integrated trace^2 infidelity = %e\n", cost);
  end # if verbose
end
