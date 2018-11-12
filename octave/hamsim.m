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
  filterTime = 1;
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
  
  Nrow = n1; # vector dimension

# estimate largest eigenvalue

  lambda = eig(H);
  maxeig = norm(lambda,"inf");
  if (verbose)
    printf("(maxeig) = (%e)\n", maxeig);
  end

# final time
  Tfinal = 50;

# time step
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*H1 estimated by maxeig
  nsteps = ceil(Tfinal/dt)+1;
  dt = Tfinal/(nsteps-1);
  if (verbose)
    printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   Tfinal, nsteps, maxeig, cfl, dt);
  end

# allocation of solution vectors
  uTime = zeros(Nrow,nsteps);

# initial data 
  uTime(:,1) = bvec;
  
# 2nd order Taylor expansion to t = +dt

  uTime(:,2) = bvec + dt * I * H *bvec - 0.5*dt^2 * (H*H*bvec) ;
#  uTime(:,2) = bvec + dt * I * H *bvec - 0.5*dt^2 * (H*H*bvec) - I*(dt^3)/6 * H*H*H*bvec ;
# adding another term in the expansion generates more wiggles

# for computing the energy
  v1 = zeros(Nrow,1);
  v2 = zeros(Nrow,1);

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

  Tperiod = Tfinal+dt;
  df = 1/Tperiod;
  Nf = nsteps;
  td = dt*[0:nsteps-1];

# Filter the numerical solution to get rid of small high wavenumber oscillations
% optionally filter
  if (filterTime)
    fc = 0.5*Nf;
    [b a]=mybutter2(2*dt*fc);
    uFilt = zeros(Nrow,nsteps);
    for k=1:Nrow
      uFilt(k,:)= myfiltfilt(b,a,uTime(k,:));
    end

    figure(3);
    h = plot(td, uTime(1,:),'b', td, uFilt(1,:),'r--');
    set(h,"linewidth",2);
    title("Time domain");
    legend("leap-frog", "Filtered");
    uTime = uFilt;
  end
  

	     # Window the time response before Fourier transforming it
#  wind = ones(1,nsteps);
#  wind = sin(pi*td/Tperiod);
  wind = exp(-( (td-0.5*Tperiod)/(0.5*Tperiod/4) ).^2);
  uWind = zeros(Nrow, nsteps);
  for (k=1:Nrow)
    uWind(k,:) = uTime(k,:).*wind;
  end
  
  freq=[ -(ceil((Nf-1)/2):-1:1), 0, (1:floor((Nf-1)/2)) ] * df; # In Hz
  om = 2*pi*freq; % angular frequency
# Discrete Fourier transformation of the time traces
#  uOmega = fftshift(fft(ifftshift(uWind),Nf,2))/Nf;
  uOmega = fftshift(fft(uWind,Nf,2))/Nf;

				# find the eigenvalues
  uAvg = sqrt(sumsq(uOmega, 1)/Nrow);
  
  if (verbose)
    figure(1);
    h = plot(td,wind,'k',td, uWind(1,:),'b', td, uWind(2,:),'r--');
    set(h,"linewidth",2);
    title("Time domain");
    legend("window","u0", "u1","location","northeast");

    figure(2);
    h = semilogy( om, uAvg, 'm+');
    
    set(h,"linewidth",2);
    set(h,"markersize",3);
    legend("SumSq(uOmega)");
    title("Frequency domain");
    
  end

end
