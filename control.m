%-*-octave-*--
%
% control: solve a model problem from quantum control theory
%
% USAGE:
% 
% control()
%
% INPUT:
%
function [cost, usave] = control(amp, verbose, cfl)
  
  if nargin < 1
    amp = 1.0;
  end

  if nargin < 2
    verbose=0;
  end

  if nargin < 3
    cfl = 0.25;
  end

  N = 4; # vector dimension
  
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

  H0 = diag([d0, d1, d2, d3]);

  H1 = [0, 1, 0, 0; 1, 0, sqrt(2), 0; 0, sqrt(2), 0, sqrt(3); 0, 0, sqrt(3), 0];

				# estimate largest eigenvalue
  H=H0+amp*H1;
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

  cfunc = zeros(1,N);
  beta = zeros(1,N);
				# Final time T
  T = 20;
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*H1 estimated by maxeig
  nsteps = ceil(T/dt);
  dt = T/nsteps;
  printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	 T, nsteps, maxeig, cfl, dt);

				# initial data and allocation of solution vectors
#  u = [0;1;0;0];
  u = U0;
  
# Taylor expansion to t = -dt ( assumes H0 is indep of t)
  t = 0;
# TODO:
# try increasing the smoothness of poly
  ## poly = amp*(3*(t/T)^2 - 2*(t/T)^3);
  ## dpoly = amp*(6*t/T^2 - 6*t^2/T^3); # = 0 for t=0; makes no difference
# 5th order
  poly = amp*(10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
  dpoly = amp/T*(30*(t/T)^2 - 60*(t/T)^3 + 30*(t/T)^4); # = 0 for t=0; makes no difference

  um = u - dt * I*H0*u + 0.5*dt^2 * (-H0*H0*u);
	 # adding another term in the expansion generates more wiggles
  d2poly = amp*(6/T^2 - 12*t/T^3); 
#  um = u - dt * I*H0*u + 0.5*dt^2 * (-H0*H0*u) - dt^3/6 * I*(-H0*H0*H0*u + d2poly*H1*u)

  ## up = zeros(N,1);
  ## v1 = zeros(N,1);
  ## v2 = zeros(N,1);
  up = zeros(N,N);
  v1 = zeros(N,1);
  v2 = zeros(N,1);
# for computing the cost function
  delta = zeros(N,1);
				# time stepping loop, harmonic oscillator
#  usave = zeros(4,nsteps+1);
#  usave(:,1) = u;
  if (verbose)
    usave = zeros(N,N,nsteps+1);
    usave(:,:,1) = u;
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
#    poly = amp*(3*(t/T)^2 - 2*(t/T)^3);
    poly = amp*(10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    up = um + 2*dt*I*(H0*u + poly*H1*u);
# cycle variables
    um = u;
    u = up;
    if (verbose)
      usave(:,:,step+1) = u;
    end
    t = t+dt;

    # accumulate cost function
    wgh = (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5);
    for c=1:N
      delta = (abs(u(:,c)).^2 - Utarget(:,c).^2 ).^2;
      beta(c) = dt*wgh*sum(delta);
    end
    cfunc = cfunc + beta; 
    
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

				# plot results
  if (verbose)
    plotunitary(usave);
  end

				# total cost function
  cost = sum(cfunc);
  printf("Cost function components: [%e, %e, %e, %e]\n", cfunc(1), cfunc(2), cfunc(3), cfunc(4));
end
