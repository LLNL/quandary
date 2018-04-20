%-*-octave-*--
%
% control: solve a model problem from quantum control theory
%
% USAGE:
% 
% [cost, dcda] = control(a1, verbose, cfl)
%
% INPUT:
% a1: amplitude of control function #1 (default = 1.0)
% verbose: 1 for plotting of the resonse (default = 0)
% cfl: CFL-number for time stepping (default = 0.25, stable for cfl<1)
%
% OUTPUT:
% cost: sum(cfunc): cost functional of the response
% dcda: derivative of cost function wrt each parameter in a1
% usave: Time histories of the 4 components of the wave function for the 4 initial data
%
function [cost, dcda_adj, ptot] = control(a1, verbose, cfl)

  abs_or_real=0; # plot the abs of the solution (1 for real)
  
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
  ## d0 = 0;
  ## d1 = 0;
  ## d2 = -1.41104006;
  ## d3 = -4.23312017;
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

				# transformation matrix
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  ## r0 = 0;
  ## r1 = d1;
  ## r2 = d2;
  ## r3 = d3;
  
  R = diag([r0, r1, r2, r3]);
  
  H0 = diag([d0, d1, d2, d3]) - R;

  H1 = [0, 1, 0, 0;
	1, 0, sqrt(2), 0;
	0, sqrt(2), 0, sqrt(3);
	0, 0, sqrt(3), 0];

# final time
  T = 20;
# first evaluate the polynomials on a coarse grid
  pad0 = timefunc(D, 100);
  ptot0 = pad0*a1;
  [pmax imax] = max(ptot0);
  [pmin imin] = min(ptot0);

# estimate largest eigenvalue
  t = (imax-1)/100 * T;
  Hc = diag([exp(-I*r0*t), exp(-I*r1*t), exp(-I*r2*t), exp(-I*r3*t)]) * H1 * ...
       diag([exp(I*r0*t), exp(I*r1*t), exp(I*r2*t), exp(I*r3*t)]);

  H=H0+pmax*Hc;
  lambda = eig(H);
  maxeig1 = norm(lambda,"inf");
  if (verbose)
    printf("(t, pmax, maxeig) = (%e, %e, %e)\n", t, pmax, maxeig1);
  end

  t = (imin-1)/100 * T;
  Hc = diag([exp(-I*r0*t), exp(-I*r1*t), exp(-I*r2*t), exp(-I*r3*t)]) * H1 * ...
       diag([exp(I*r0*t), exp(I*r1*t), exp(I*r2*t), exp(I*r3*t)]);

  H=H0+pmin*Hc;
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

				# Target state at t=20
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
  printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	 T, nsteps, maxeig, cfl, dt);

		# evaluate the polynomials at the discrete time levels
  td = linspace(0,T,nsteps+1)'; # column vector
# evaluate all polynomials on the grid
  pad = timefunc(D, nsteps);
#  pad(:, 1) = (10*(td./T).^3 - 15*(td./T).^4 + 6*(td./T).^5); # first polynomial
# sum up all polynomial components
  ptot = pad*a1;

# form the weight function
  tp = 0.5*T;
  t0 = T;
  tau = (td - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);
  wghf = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;
# old weight function
#  wghf = (10*(td/T).^3 - 15*(td/T).^4 + 6*(td/T).^5);

				# initial data and allocation of solution vectors
  u = U0;
  
# Taylor expansion to t = -dt ( assumes H0 is indep of t)

# the Hc term doesn't make any difference because ptot(1)=0
  Hc = diag([exp(-I*r0*t), exp(-I*r1*t), exp(-I*r2*t), exp(-I*r3*t)]) * H1 * ...
       diag([exp(I*r0*t), exp(I*r1*t), exp(I*r2*t), exp(I*r3*t)]);

  um = u - dt * I*(H0)*u + 0.5*dt^2 * (-(H0)*(H0)*u);
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
    pval = ptot(step);
    Hc = diag([exp(-I*r0*t), exp(-I*r1*t), exp(-I*r2*t), exp(-I*r3*t)]) * H1 * ...
	 diag([exp(I*r0*t), exp(I*r1*t), exp(I*r2*t), exp(I*r3*t)]);
    up = um + 2*dt*I*(H0*u + pval*Hc*u);
# sensitivity wrt parameter D
    pa = pad(step, D);
    uap = uam + 2*dt*I*(H0*ua + pval*Hc*ua + pa*Hc*u); 

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
#    wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    for c=1:N
      delta = (abs(u(:,c)).^2 - Utarget(:,c).^2 ).^2;
      beta(c) = dt*wghf(step+1)*sum(delta);
    end
    cfunc = cfunc + beta; 
# sensitivity
    for c=1:N
      cvect = wghf(step+1)* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
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
    plotunitary(usave, abs_or_real);
    
    figure(5);
    subplot(2,1,1);
    h=plot(td, ptot);
    set(h,"linewidth",2);
    title("Forcing function");

    subplot(2,1,2);
    h = plot(td, wghf);
    set(h,"linewidth",2);
    title("Weight function");

  end

				# total cost function
  cost = sum(cfunc);
  dcda = sum(ca1);
  printf("Forward calculation: Parameter a1 =[ %e", a1(1));
  for q=2:D
    printf(", %e", a1(q));
  end
  printf(" ]\n");
  printf("Total cost function c = %e, sum[%e, %e, %e, %e]\n", cost, cfunc(1), cfunc(2), cfunc(3), cfunc(4));
  printf("dcda(%d) = %e, sum[%e, %e, %e, %e]\n", D, dcda, ca1(1), ca1(2), ca1(3), ca1(4));

# now compute the derivative of the cost function by solving the forwards problem
# backwards together with the adjoint problem
# Terminal conditions are up = psi(T) and u = psi(T-dt): Bot are given by the above solution of the forwards problem

# homogeneous terminal conditions for the adjoint wave function
  lap   = zeros(N,N);
# forcing for adjoint eqn at t=T
  adf = zeros(N,N);
  # forcing for the adjoint equation depends on u=psi(t)
  t = T;
  for c=1:N
    adf(:,c) = 4*wghf(nsteps+1)* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
  end
# improve compatibility of terminal conditions. Is this correct?
  la = dt*adf;
  
# cycle local arrays to get ready for reverse time stepping  
  up = u;
  u = um;

  ca1_adj = zeros(N,D);
  apla_sp = zeros(N,D);

  ## if (verbose)
  ##   ursave = zeros(N,N,nsteps+1);
  ##   ursave(:,:,nsteps+1) = lap;
  ##   ursave(:,:,nsteps) = la;
  ## end
  
  t=T-dt;
  Hc = diag([exp(-I*r0*t), exp(-I*r1*t), exp(-I*r2*t), exp(-I*r3*t)]) * H1 * ...
       diag([exp(I*r0*t), exp(I*r1*t), exp(I*r2*t), exp(I*r3*t)]);

# sensitivity (no contribution from t=T because lambda(T)=0
  for q=1:D
    pa = pad(nsteps, q); # T - dt
    amat_psi = I * pa * Hc * u; 

    for c=1:N
      apla_sp(c,q) = dt*real(dot(amat_psi(:,c), la(:,c)));
    end
  end #for
  ca1_adj = ca1_adj + apla_sp;
  
# backwards time stepping  
  for step=nsteps-1:-1:1
    pval = ptot(step+1);
    um = up - 2*dt*I*(H0*u + pval*Hc*u); # Hc(T-dt) computed above for the first step
# forcing for the adjoint equation depends on u=psi(t)
    wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    for c=1:N
      adf(:,c) = 4*wghf(step+1)* u(:,c).*( abs(u(:,c)).^2 - Utarget(:,c).^2 );
    end
# evolve the adjoint eqn
    lam = lap - 2*dt*I*(H0*la + pval*Hc*la) + 2*dt*adf;
    
# cycle variables
    up = u;
    u = um;
    lap = la;
    la = lam;

    ## if (verbose)
    ##   ursave(:,:,step) = la;
    ## end
    t = t-dt;

# update Hc
    Hc = diag([exp(-I*r0*t), exp(-I*r1*t), exp(-I*r2*t), exp(-I*r3*t)]) * H1 * ...
	 diag([exp(I*r0*t), exp(I*r1*t), exp(I*r2*t), exp(I*r3*t)]);

# sensitivity:
    for q=1:D
      pa = pad(step, q);
      amat_psi = I * pa * Hc * u; 

      for c=1:N
	apla_sp(c,q) = dt*real(dot(amat_psi(:,c), la(:,c)));
      end
    end #for
    ca1_adj = ca1_adj + apla_sp;
    
  end # backwards time stepping loop

  ca1_adj = ca1_adj - 0.5*apla_sp; 
  dcda_adj = sum(ca1_adj,1);
  
  printf("Adjoint calculation:\n");
  for q=1:D
    printf("dcda(%d) = %e, sum[%e, %e, %e, %e]\n", q, dcda_adj(q), ca1_adj(1,q), ca1_adj(2,q), ca1_adj(3,q), ca1_adj(4,q));
  end

end
