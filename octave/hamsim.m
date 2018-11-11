%-*-octave-*--
%
% hamsim: perform a Hamiltonian simulation u(t) = exp(iHt) b
%
% USAGE:
% 
% [u] = hamsim(H, bvec, verbose, cfl)
%
% INPUT:
% H: Hermitian matrix
% bvec: Right hand side
%
% OUTPUT:
% u: solution vector at every time step
%
function [uTime] = hamsim(H, bvec, verbose, cfl)

  abs_or_real=0; # plot the abs of the solution (1 for real)

  if nargin < 1
    H = [4.5,-5.5; -5.5, 4.5];
  end

  if nargin < 2
    bvec = [2;1];
  end

  if nargin < 3
    verbose = 1;
  end
  
  if nargin < 4
    cfl = 0.25;
  end

  [n1 n2] = size(H);
  [n3 n4] = size(bvec);
  printf("size(H)=(%d, %d), size(bvec)=(%d, %d)\n", n1, n2, n3, n4);

% The number of columns in b doesn't really matter
  if (n1 == n2 && n3 == n1)
    goodDim=1;
  else
    goodDim=0;
  end

  if !goodDim
    printf("Dimensions make no sense, sorry...\n");
    return;
  end
  printf("Dimensions look ok...\n");
#  return;
  
  N = n1; # vector dimension

# final time
  T = 50;

# estimate largest eigenvalue

  lambda = eig(H);
  maxeig = norm(lambda,"inf");
  if (verbose)
    printf("(maxeig) = (%e)\n", maxeig);
  end

  cfunc = zeros(N,1);
  beta = zeros(N,1);
  cu_sp = zeros(N,1);
  ca1 = zeros(N,1);
  vect = zeros(N,1);
				# Final time T
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*H1 estimated by maxeig
  nsteps = ceil(T/dt)+1;
  dt = T/(nsteps-1);
  if (verbose)
    printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   T, nsteps, maxeig, cfl, dt);
  end

# allocation of solution vectors
  uTime = zeros(N,nsteps);

# initial data 
  uTime(:,1) = bvec;
  
# 2nd order Taylor expansion to t = +dt

  uTime(:,2) = uTime(:,1) + dt * I * H *uTime(:,1) + 0.5*dt^2 * (-H*H*uTime(:,2));
# adding another term in the expansion generates more wiggles

# for computing the energy
  v1 = zeros(N,1);
  v2 = zeros(N,1);

  t=dt;
  step=1;
# evaluate energy
  if (verbose)
    v1 = uTime(:,step+1) + uTime(:,step);
    v2 = uTime(:,step+1) - uTime(:,step);
    energy = 0.25*(norm(v1)^2 - norm(v2)^2);

    printf("Time step = %d, time = %e, energy = %e\n", step, t, energy);
  end # if verbose

# time stepping loop, harmonic oscillator

  for step=2:nsteps-1
    uTime(:,step+1) = uTime(:,step-1) + 2*dt*I*(H*uTime(:,step));

    t = t+dt;

# evaluate energy
    if (verbose)
      if (mod(step,1000)==0 || step==nsteps-1)
	v1 = uTime(:,step+1) + uTime(:,step);
	v2 = uTime(:,step+1) - uTime(:,step);
	energy = 0.25*(norm(v1)^2 - norm(v2)^2);

	printf("Time step = %d, time = %e, energy = %e\n", step, t, energy);
      end
    end # if verbose
  end # time stepping loop

  df = 1/T;
  Nf = nsteps;
  freq=[ -(ceil((Nf-1)/2):-1:1), 0, (1:floor((Nf-1)/2)) ] * df; # In Hz
  om = 2*pi*freq; % angular frequency
# Discrete Fourier transformation of the time traces
  uOmega = fftshift(fft(ifftshift(uTime),Nf,2));
  
  if (verbose)
    figure(1);
    td = dt*[0:nsteps-1];
    h = plot(td, uTime(1,:),'b', td, uTime(2,:),'r');
    set(h,"linewidth",2);
    title("Time domain");
    legend("u0", "u1","location","north");

    figure(2);
    h = plot( om, abs(uOmega(1,:)), 'm+',  om, abs(uOmega(2,:)), 'cd');
    set(h,"linewidth",2);
    set(h,"markersize",3);
    legend("Abs(u0)", "Abs(u1)", "location", "east");
    title("Frequency domain");
    
  end

end
