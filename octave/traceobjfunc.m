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

  H1 = [0, 1, 0, 0;
  	1, 0, sqrt(2), 0;
  	0, sqrt(2), 0, sqrt(3);
  	0, 0, sqrt(3), 0];

  S1 = [0, 1, 0, 0;
  	-1, 0, sqrt(2), 0;
  	0, -sqrt(2), 0, sqrt(3);
  	0, 0, -sqrt(3), 0];

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

  H=H0+pmax*H1;
  lambda = eig(H);
  maxeig1 = norm(lambda,"inf");
  if (verbose)
    printf("(t, pmax, maxeig) = (%e, %e, %e)\n", t, pmax, maxeig1);
  end

  t = (imin-1)/100 * T;

  H=H0+pmin*H1;
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
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*H1 estimated by maxeig
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
  k1 = zeros(N,N);
  k2 = zeros(N,N);
  l1 = zeros(N,N);
  l2 = zeros(N,N);

# testing octave syntax
  step=1;
  S = qtot(step).*S1; # skew-symmetric part
  rhs = H*ur + S*vi;
  l1 = linsolve( Ident-0.5*dt*S, rhs ); # - 0.5*dt*S);
  printf("Size(Ident)= (%d, %d), size(rhs)=(%d,%d), size(l1)=(%d,%d)\n", size(Ident), size(rhs), size(l1));
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

  t=0;
  step=0;
  for step=1:nsteps
    H = H0 + ptot(step).*H1; # symmetric part
    S = qtot(step).*S1; # skew-symmetric part
    expH = expm(-I*dt*H);

    uSol = expH * uSol;

		    # Partitioned 2nd order RK method (Stromer-Verlet)
    rhs = H*ur + S*vi;
    l1 = linsolve( Ident-0.5*dt*S, rhs );
    k1 = S*ur - H*(vi+0.5*dt*l1);
    rhs = S*(ur+0.5*dt*k1) - H*(vi+0.5*dt*l1);
    k2 = linsolve( Ident-0.5*dt*S, rhs );
    l2 = H*(ur+0.5*dt*(k1+k2)) + S*(vi+0.5*dt*l1);

    ur = ur + 0.5*dt*(k1 + k2);
    vi = vi + 0.5*dt*(l1 + l2);
    t = t+dt;

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
