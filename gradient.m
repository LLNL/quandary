%-*-octave-*--
%
% gradient: compute the gradient of the objective function from quantum control theory
%
% USAGE:
% 
% [dcda] = gradient(a1)
%
% INPUT:
% a1: amplitude of control function #1 (default = 1.0)
%
% OUTPUT:
% cost: sum(cfunc): cost functional of the response
% dcda: derivative of cost function wrt each parameter in a1
%
function [dcda_adj] = gradient(a1, verbose)

  abs_or_real=0; # plot the abs of the solution (1 for real)

  wconst = 0.01;   # for response to e2 and e3

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

  H1 = [0, 1, 0, 0;
	1, 0, sqrt(2), 0;
	0, sqrt(2), 0, sqrt(3);
	0, 0, sqrt(3), 0];

# final time
  T = 15;
# first evaluate the polynomials on a coarse grid
  pad0 = timefunc(D, 100);
  ptot0 = pad0*a1;
  [pmax imax] = max(ptot0);
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
    end
  end # time stepping loop

# plot results
  if (verbose)
    plotunitary(usave, T, abs_or_real);
    
    figure(5);
    subplot(2,1,1);
    h=plot(td, ptot);
    set(h,"linewidth",2);
    title("Forcing function");

    subplot(2,1,2);
    h = plot(td, wghf(:,1), td, wghf(:,3));
    set(h,"linewidth",2);
    title("Weight function");
    legend("e0 & e1", "e2 & e3","location","north");

  end

# Compute the derivative of the cost function by solving the problem backwards,
# together with the adjoint problem
# Terminal conditions are up = psi(T) and u = psi(T-dt): Both are given by the above solution of the forwards problem

# homogeneous terminal conditions for the adjoint wave function
  lap   = zeros(N,N);
# forcing for adjoint eqn at t=T
  adf = zeros(N,N);
  # forcing for the adjoint equation depends on u=psi(t)
  t = T;
  for c=1:N
    adf(:,c) = 4*wghf(nsteps+1,c)* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
  end
# improve compatibility of terminal conditions. Is this correct?
  la = dt*adf;
  
# cycle local arrays to get ready for reverse time stepping  
  up = u;
  u = um;

  ca1_adj = zeros(N,D);
  apla_sp = zeros(N,D);

  t=T-dt;

# sensitivity (no contribution from t=T because lambda(T)=0
  for q=1:D
    pa = pad(nsteps, q); # T - dt
    amat_psi = I * pa * H1 * u; 

    for c=1:N
      apla_sp(c,q) = dt*real(dot(amat_psi(:,c), la(:,c)));
    end
  end #for
  ca1_adj = ca1_adj + apla_sp;
  
# backwards time stepping  
  for step=nsteps-1:-1:1
    pval = ptot(step+1);
    um = up - 2*dt*I*(H0*u + pval*H1*u); 
# forcing for the adjoint equation depends on u=psi(t)
    wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    for c=1:N
      adf(:,c) = 4*wghf(step+1,c)* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
    end
# evolve the adjoint eqn
    lam = lap - 2*dt*I*(H0*la + pval*H1*la) + 2*dt*adf;
    
# cycle variables
    up = u;
    u = um;
    lap = la;
    la = lam;

    t = t-dt;

# sensitivity:
    for q=1:D
      pa = pad(step, q); # evaluate control function
      amat_psi = I * pa * H1 * u; 

      for c=1:N
	apla_sp(c,q) = dt*real(dot(amat_psi(:,c), la(:,c)));
      end
    end #for
    ca1_adj = ca1_adj + apla_sp;
    
  end # backwards time stepping loop

  ca1_adj = ca1_adj - 0.5*apla_sp; 
  dcda_adj = sum(ca1_adj,1);
		      # transpose gradient to be compatible with sqp()
  dcda_adj = dcda_adj';
  
  if (verbose)
    printf("Adjoint calculation:\n");
    for q=1:D
      printf("dcda(%d) = %e, sum[%e, %e, %e, %e]\n", q, dcda_adj(q), ca1_adj(1,q), ca1_adj(2,q), ca1_adj(3,q), ca1_adj(4,q));
    end
  end # if verbose
  
end
