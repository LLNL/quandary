%-*-octave-*--
%
% control: solve a model problem from quantum control theory
%
% USAGE:
% 
% [cost, usave] = control(a1, verbose, cfl)
%
% INPUT:
% a1: amplitude of control function #1 (default = 1.0)
% verbose: 1 for plotting of the resonse (default = 0)
% cfl: CFL-number for time stepping (default = 0.25, stable for cfl<1)
%
% OUTPUT:
% cost: sum(cfunc): cost functional of the response
% usave: Time histories of the 4 components of the wave function for the 4 initial data
%
function [cost, dcda_adj, ursave] = control(a1, verbose, cfl)
  
  if nargin < 1
    a1 = 1.0;
  end

  if nargin < 2
    verbose=0;
  end

  if nargin < 3
    cfl = 0.25;
  end

  N = 4; # vector dimension

  D = length(a1); # parameter dimension
  
# coefficients in H0
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

  H0 = diag([d0, d1, d2, d3]);

  H1 = [0, 1, 0, 0; 1, 0, sqrt(2), 0; 0, sqrt(2), 0, sqrt(3); 0, 0, sqrt(3), 0];

# first evaluate the polynomials on a coarse grid
  pad0 = timefunc(D, 100);
  ptot0 = pad0*a1;
  pmax=max(abs(ptot0));

# estimate largest eigenvalue
  H=H0+pmax*H1;
  lambda = eig(H);
  maxeig = norm(lambda,"inf");

# the basis for the initial data as a matrix
  U0=diag([1, 1, 1, 1]);

				# Target state at t=20
  Utarget = ...
  [0, 1, 0, 0; ...
   1, 0, 0, 0; ...
   0, 0, 1, 0; ...
   0, 0, 0, 1];

  cfunc = zeros(N,1);
  beta = zeros(N,1);
  cu_sp = zeros(N,1);
  ca1 = zeros(N,1);
  vect = zeros(N,1);
				# Final time T
  T = 20;
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*H1 estimated by maxeig
  nsteps = ceil(T/dt);
  dt = T/nsteps;
  printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	 T, nsteps, maxeig, cfl, dt);

		# evaluate the polynomials at the discrete time levels
  td = linspace(0,T,nsteps+1)'; # column vector
# evaluate all polynomials on the grid
  pad = timefunc(D, nsteps);
#  pad(:, 1) = (10*(td./T).^3 - 15*(td./T).^4 + 6*(td./T).^5); # first polynomial
# sum up all polynomial components
  ptot = pad*a1;
  
				# initial data and allocation of solution vectors
  u = U0;
  
# Taylor expansion to t = -dt ( assumes H0 is indep of t)
  t = 0;

  um = u - dt * I*H0*u + 0.5*dt^2 * (-H0*H0*u);
# adding another term in the expansion generates more wiggles

  up = zeros(N,N);
  v1 = zeros(N,1);
  v2 = zeros(N,1);
# for computing the cost function
  delta = zeros(N,1);

# solve for du/da (satisfies homogeneous initial conditions)
  uap  = zeros(N,N);
  ua    = zeros(N,N);
  uam = zeros(N,N);
  ## v1 = zeros(N,1);
  ## v2 = zeros(N,1);

			     # time stepping loop, harmonic oscillator
  if (verbose)
    usave = zeros(N,N,nsteps+1);
    usave(:,:,1) = u;
    uasave = zeros(N,N,nsteps+1);
    uasave(:,:,1) = ua;
  end

  t=0;
  step=0;
  energy = 0;
  for c=1:N
    v1 = u(:,c) + um(:,c);
    v2 = u(:,c) - um(:,c);
    energy = energy + 0.25*(norm(v1)^2 - norm(v2)^2);
  end
  printf("Time step = %d, time = %e, energy = %e\n", step, t, energy/4);
# cost function doesn't get a contribution from the initial data because the weight is zero
  for step=1:nsteps
#    pval = a1*(10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    pval = ptot(step);
    up = um + 2*dt*I*(H0*u + pval*H1*u);
# sensitivity
#    pa = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    pa = pad(step, 1);
    uap = uam + 2*dt*I*(H0*ua + pval*H1*ua + pa*H1*u); 

# cycle variables
    um = u;
    u = up;
    uam = ua;
    ua = uap;
    if (verbose)
      usave(:,:,step+1) = u;
      uasave(:,:,step+1) = ua;
    end
    t = t+dt;

    # accumulate cost function
    wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    for c=1:N
      delta = (abs(u(:,c)).^2 - Utarget(:,c).^2 ).^2;
      beta(c) = dt*wgh*sum(delta);
    end
    cfunc = cfunc + beta; 
# sensitivity
    for c=1:N
      cvect = wgh* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
      cu_sp(c) = 4*dt*real(dot(cvect, ua(:,c)));
    end
    ca1 = ca1 + cu_sp;
# evaluate energy
    if (mod(step,1000)==0 || step==nsteps)
      energy = 0;
      for (c=1:4)
	v1 = u(:,c) + um(:,c);
	v2 = u(:,c) - um(:,c);
	energy = energy + 0.25*(norm(v1)^2 - norm(v2)^2);
      end
      printf("Time step = %d, time = %e, energy = %e\n", step, t, energy/4);
    end
  end # time stepping loop

  # subtract out 1/2 of last contribution for 2nd order accuracy
  cfunc = cfunc - 0.5*beta; 
  ca1 = ca1 - 0.5*cu_sp; 

				# plot results
  if (verbose)
    plotunitary(usave);
  end

				# total cost function
  cost = sum(cfunc);
  dcda = sum(ca1);
  printf("Forward calculation: Parameter:\n");
  a1
  printf("Total cost function c = %e, dc/da1 = %e\n", cost, dcda);
  printf("Cost function components: [%e, %e, %e, %e]\n", ...
	 cfunc(1), cfunc(2), cfunc(3), cfunc(4));
  printf("dc/da1 components = [%e, %e, %e, %e]\n", ca1(1), ca1(2), ca1(3), ca1(4));

# now compute the derivative of the cost function by solving the forwards problem
# backwards together with the adjoint problem
# Terminal conditions are up = psi(T) and u = psi(T-dt): Bot are given by the above solution of the forwards problem

# homogeneous terminal conditions for the adjoint wave function
  lap   = zeros(N,N);
# forcing for adjoint eqn at t=T
  adf = zeros(N,N);
  # forcing for the adjoint equation depends on u=psi(t)
  t = T;
  wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
  for c=1:N
    adf(:,c) = 4*wgh* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
  end
# improve compatibility of terminal conditions. Is this correct?
  la = dt*adf;
#  la = zeros(N,N);
  
# cycle local arrays to get ready for reverse time stepping  
  up = u;
  u = um;

  ca1_adj = zeros(N,1);

  if (verbose)
    ursave = zeros(N,N,nsteps+1);
    ursave(:,:,nsteps+1) = lap;
    ursave(:,:,nsteps) = la;
  end
  
  t=T-dt;
# sensitivity (no contribution from t=T because lambda(T)=0
#  pa = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5); 
  pa = pad(nsteps, 1); # T - dt
  amat_psi = I * pa * H1 * u; 

  for c=1:N
    cu_sp(c) = dt*real(dot(amat_psi(:,c), la(:,c)));
  end

  ca1_adj = ca1_adj + cu_sp;
  
# backwards time stepping  
  for step=nsteps-1:-1:1
#    pval = a1*(10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    pval = ptot(step+1);
    um = up - 2*dt*I*(H0*u + pval*H1*u);
# forcing for the adjoint equation depends on u=psi(t)
    wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    for c=1:N
      adf(:,c) = 4*wgh* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
    end
# evolve the adjoint eqn
    lam = lap - 2*dt*I*(H0*la + pval*H1*la) + 2*dt*adf;
    
# cycle variables
    up = u;
    u = um;
    lap = la;
    la = lam;

    if (verbose)
      ursave(:,:,step) = la;
    end
    t = t-dt;

# sensitivity
#    pa = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    pa = pad(step, 1);
    amat_psi = I * pa * H1 * u; 

    for c=1:N
      cu_sp(c) = dt*real(dot(amat_psi(:,c), la(:,c)));
    end

    ca1_adj = ca1_adj + cu_sp;
    
  end # backwards time stepping loop

  ca1_adj = ca1_adj - 0.5*cu_sp; 
  dcda_adj = sum(ca1_adj);
  
  printf("Adjoint calculation:\n");
  printf("dc/da1 components = [%e, %e, %e, %e]\n", ca1_adj(1), ca1_adj(2), ca1_adj(3), ca1_adj(4));
  
end
