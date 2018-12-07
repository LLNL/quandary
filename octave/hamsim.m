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
% xvec: approximate solution of H xvec = bvec
%
function [xvec] = hamsim(H, bvec, verbose, cfl)
  eps = 1e-5;
  firstOrder = 1;
  leapFrog = 0;
  magnus = 1;
  rk4 = 0;
  abs_or_real=0; # plot the abs of the solution (1 for real)

#  U=[1, 0; 0, 1];
  U=[1, 1; -1, 1]/sqrt(2);
#  Lambda=2*pi*[1 , 0; 0, 2];
  Lambda=[1 , 0; 0, 2];

  # final time
  Tper = 100*2*pi;

  H = U*Lambda*U';

  H2 = H*H;

  if nargin < 2
    bvec = [1;2];
  end

  if nargin < 3
    verbose = 1;
  end
  
  if nargin < 4
    cfl = 0.25; #depends on the method
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

# time step
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*H1 estimated by maxeig
  nsteps = ceil(Tper/dt);
  dt = Tper/(nsteps);
  if (verbose)
    printf("Time period = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   Tper, nsteps, maxeig, cfl, dt);
  end

# allocation of solution vectors
  uTime = zeros(Nrow,nsteps);

# initial data 
  uTime(:,1) = bvec;
  
# 3rd order Taylor expansion to t = +dt
#  uTime(:,2) = bvec + dt * I * H *bvec - 0.5*dt^2 * (H2*bvec) ;

# 4th order Taylor expansion to t = +dt
  uTime(:,2) = bvec + dt * I * H *bvec - 0.5*dt^2 * (H*H*bvec) - I*(dt^3)/6 * H*H*H*bvec ;
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
    energy1 = 0.25*(norm(v1)^2 - norm(v2)^2);
    energy = uTime(:,1)'*H*uTime(:,1);

    printf("Time step = %d, time = %e, energy1 = %e, energy = %e\n", step, t, energy1, energy);
  end # if verbose

# time stepping loop, harmonic oscillator, 1st order formulation

  if (firstOrder)
    if (magnus)
       expH = zeros(Nrow,Nrow);
      ## for row=1:Nrow
      ## 	expH(row,row) = exp(I*dt*Lambda(row,row));
      ## end
      expH = expm(I*dt*H);
      for step=1:nsteps-1
	uTime(:,step+1) = expH * uTime(:,step);

	t = t+dt;
				# evaluate energy
	if (verbose)
	  if (mod(step,1000)==0 || step==nsteps-1)
	    v1 = uTime(:,step+1) + uTime(:,step);
	    v2 = uTime(:,step+1) - uTime(:,step);
	    energy = 0.25*(norm(v1)^2 - norm(v2)^2);
	    energy = uTime(:,step+1)'*H*uTime(:,step+1);

	    printf("Time step = %d, time = %e, energy = %e\n", step, t, energy);
	  end
	end # if verbose
      end # time stepping loop
    else
      printf("ERROR: Undefined time-integration method\n");
      return;
    end
  else
# second order formulation
    printf("ERROR: Undefined time-integration method\n");
  end
  
  df = 1/Tper;
  Nf = nsteps;
  td = dt*[0:nsteps-1];

#  legend("u0", "u1","location","northeast");

	     # Window the time response before Fourier transforming it
  wind = ones(1,nsteps);
#  wind = sin(pi*td/Tperiod);
#  wind = exp(-( (td-0.5*Tperiod)/(0.5*Tperiod/4) ).^2);
  uWind = zeros(Nrow, nsteps);
  for (k=1:Nrow)
    uWind(k,:) = uTime(k,:).*wind;
  end
  
  freq=[ -(ceil((Nf-1)/2):-1:1), 0, (1:floor((Nf-1)/2)) ] * df; # In Hz
  om = 2*pi*freq; % angular frequency
# Discrete Fourier transformation of the time traces
  uOmega = fftshift(fft(ifftshift(uWind),Nf,2))/Nf;

				# find the eigenvalues
  uAvg = sqrt(sumsq(uOmega, 1)/Nrow);
  maxPnts=zeros(Nrow,1);
  numMaxPnts=0;
  for k=2:nsteps-1
    dp = uAvg(k+1) - uAvg(k);
    dm = uAvg(k) - uAvg(k-1);
    if (uAvg(k) > eps && dp*dm < 0 && dm>0)
      numMaxPnts = numMaxPnts+1;
      maxPnts(numMaxPnts)=k;
    end
  end

  xvec = zeros(n1,1);
  for q=1:numMaxPnts
    eval = om(maxPnts(q));
    printf("Max at q=%d, freq(q)=%e\n", maxPnts(q), om(maxPnts(q)));
    uHat = uOmega(:,maxPnts(q));
    printf("Fouier mode: ");
    uHat
				# accumulate solution
    xvec = xvec + uHat./eval;
  end

  printf("Approximate solution:");
  xvec
    
  printf("Exact solution:");
  xsol = H\bvec

  figure(3);
  h = semilogy( om, uAvg, 'm-');
    
  set(h,"linewidth",1.5);
  set(h,"markersize",3);
  legend("SumSq(uOmega)");
  title("Fourier magnitude");
  xlabel("Angular frequency");
  
				# try to reconstruct the time function
  ## uRec=zeros(Nrow, nsteps);
  ## for q=1:numMaxPnts
  ##   lambda = om(maxPnts(q));
  ##   uHat = uOmega(:, maxPnts(q));
  ##   for row=1:Nrow
  ##     uRec(row,:) = uRec(row,:) + uHat(row)*exp(I*lambda*td);
  ##   end
  ## end
  
  ## figure(1);
  ## h = plot(td, wind, 'k', td, real(uTime(1,:)),'b', td, imag(uTime(1,:)),'r',...
  ## 	   td, real(uRec(1,:)),'c--', td, imag(uRec(1,:)),'m--');
  ## set(h,"linewidth",1.5);
  ## set(h,"markersize",3);
  ## title("1st comp, real+imag part, orig+recon, time domain");

  ## figure(2);
  ## h = plot(td, wind, 'k', td, real(uTime(2,:)),'b', td, imag(uTime(2,:)),'r',...
  ## 	   td, real(uRec(2,:)),'c--', td, imag(uRec(2,:)),'m--');
  ## set(h,"linewidth",1.5);
  ## set(h,"markersize",3);
  ## title("2nd comp, real+imag part, orig+recon, time domain");

 end
