%-*-octave-*--
%
% objective: solve a model problem from quantum control theory
%
% USAGE:
% 
% [cost, ufinal] = objective(a1, verbose, cfl)
%
% INPUT:
% a1: amplitude of control function #1 (default = 1.0)
%
% OUTPUT:
% cost: sum(cfunc): cost functional of the response
% ufinal: state vector at t=T
%
function [cost ufinal] = objective(a1, verbose)

  abs_or_real=0; # plot the abs of the solution (1 for real)

  wconst = 0.01; # for response to e2 and e3
  
  if nargin < 1
    a1 = 1.0;
  end

  if nargin < 2
    verbose=0;
  end

  cfl = 0.25;

  N = 4; # vector dimension

  D = length(a1); # parameter dimension
  
# coefficients in H0
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

  H0 = diag([d0, d1, d2, d3]);

  ## H1 = I*[0, 1, 0, 0;
  ## 	  -1, 0, sqrt(2), 0;
  ## 	  0, -sqrt(2), 0, sqrt(3);
  ## 	  0, 0, -sqrt(3), 0];

  H1 = [0, 1, 0, 0;
  	1, 0, sqrt(2), 0;
  	0, sqrt(2), 0, sqrt(3);
  	0, 0, sqrt(3), 0];

# final time
  T = 15;
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
  
# the basis for the initial data as a matrix
  U0=diag([1, 1, 1, 1]);

				# Target state at t=T
  Utarget = [0, 1, 0, 0;    1, 0, 0, 0;   0, 0, 1, 0;  0, 0, 0, 1];

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
  td = linspace(0,T,nsteps+1)'; # column vector
# evaluate all polynomials on the grid
  pad = timefunc(D, nsteps);
#  pad(:, 1) = (10*(td./T).^3 - 15*(td./T).^4 + 6*(td./T).^5); # first polynomial
# sum up all polynomial components
  ptot = pad*a1;

# form the weight function
  tp = 0.125*T;
  t0 = T;
  tau = (td - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);
  wghf1 = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;

# different weight functions for different components
  wghf = zeros(nsteps+1,N);
  wghf(:,1) = wghf1;
  wghf(:,2) = wghf1;
  wghf(:,3) = (1-wconst)*wghf1+wconst;
  wghf(:,4) = (1-wconst)*wghf1+wconst;

# initial data and allocation of solution vectors
  u = U0;
  
# Taylor expansion to t = -dt ( assumes H0 is indep of t)

# the H1 term doesn't make any difference because ptot(1)=0

  um = u - dt * I*(H0)*u + 0.5*dt^2 * (-(H0)*(H0)*u);
# adding another term in the expansion generates more wiggles

  up = zeros(N,N);
  v1 = zeros(N,1);
  v2 = zeros(N,1);
# for computing the cost function
  delta = zeros(N,1);

			     # time stepping loop, harmonic oscillator
  if (verbose)
    usave = zeros(N,N,nsteps+1);
    usave(:,:,1) = u;
  end

  t=0;
  step=0;
  if (verbose)
    energy = 0;
    for c=1:N
      v1 = u(:,c) + um(:,c);
      v2 = u(:,c) - um(:,c);
      energy = energy + 0.25*(norm(v1)^2 - norm(v2)^2);
    end
    printf("Time step = %d, time = %e, energy = %e\n", step, t, energy/4);
  end
# cost function doesn't get a contribution from the initial data because the weight is zero
  for step=1:nsteps
    pval = ptot(step);
    up = um + 2*dt*I*(H0*u + pval*H1*u);

# cycle variables
    um = u;
    u = up;
    if (verbose)
      usave(:,:,step+1) = u;
    end
    t = t+dt;

    # accumulate cost function
    for c=1:N
      delta = (abs(u(:,c)).^2 - Utarget(:,c).^2 ).^2;
      beta(c) = dt*wghf(step+1,c)*sum(delta);
    end
    cfunc = cfunc + beta; 

# evaluate energy
    if (verbose)
      if (mod(step,1000)==0 || step==nsteps)
	energy = 0;
	for (c=1:4)
	  v1 = u(:,c) + um(:,c);
	  v2 = u(:,c) - um(:,c);
	  energy = energy + 0.25*(norm(v1)^2 - norm(v2)^2);
	end
	printf("Time step = %d, time = %e, energy = %e\n", step, t, energy/4);
      end
    end # if verbose
  end # time stepping loop

  # subtract out 1/2 of last contribution for 2nd order accuracy
  cfunc = cfunc - 0.5*beta; 
  ca1 = ca1 - 0.5*cu_sp; 

				# save final state
  ufinal = u;
				# plot results
  if (verbose)
    plotunitary(usave, T, abs_or_real);
    
    figure(5);
    subplot(2,1,1);
    h=plot(td, ptot);
    pmin = min(ptot);
    pmax = max(ptot);
    pdelta=pmax-pmin;
    axis([0,T,pmin-0.1*pdelta,pmax+0.1*pdelta]);
    set(h,"linewidth",2);
    title("Forcing function");

    subplot(2,1,2);
#    h = plot(td, wghf(:,1), td, wghf(:,3));
#    axis tight;
#    set(h,"linewidth",2);
#    title("Weight function");
#    legend("e0 & e1", "e2 & e3","location","north");
    h = plot( [1:D/2], a1(1:2:end), "b*",  [1:D/2], a1(2:2:end), "r*");
    set(h,"markersize",10);
    amin = min(a1);
    amax= max(a1);
    adelta=amax-amin;
    axis([0.5 D/2+0.5, amin-0.1*adelta, amax+0.1*adelta]);
    legend("cos", "sin", "location", "east");
    title("Parameters");
  end

				# total cost function
  cost = sum(cfunc);
  if (verbose)
    printf("Forward calculation: Parameter a1 =[ %e", a1(1));
    for q=2:D
      printf(", %e", a1(q));
    end
    printf(" ]\n");
    printf("Total cost function c = %e, sum[%e, %e, %e, %e]\n", cost, cfunc(1), cfunc(2), cfunc(3), cfunc(4));
  end # if verbose
end
