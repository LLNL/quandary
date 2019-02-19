%-*-octave-*--
%
% traceobjf1: evaluate the objective function using the trace norm
%
% USAGE:
% 
% [objF, uFinal] = traceobjf1(pcof, order, verbose)
%
% INPUT:
% pcof(D,1): amplitudes of the control functions as a D x 1 column vector, D=size(pcof,1)
% verbose: 0: quite mode, 1: verbose
% order: order of accuracy: 2, 4, or 6.
%
% OUTPUT:
% objF: trace norm of gate infidelity cost functional
% uFinal_r: Real part of state vector at t=T
% uFinal_i: Imaginary part of state vector at t=T
%
function [objf_v, uFinal_r, uFinal_i] = traceobjf1(pcof, order, verbose)

  N = 4; # vector dimension
  Nguard = 3; # number of extra levels
  T=30;# final time
  xi=1/N; # coefficient for penalizing forbidden states
  abs_or_real=0; # plot the magnitude (abs) of real part of the solution (1 for real)

  if nargin < 1
    pcof(1) = 1.0;
    pcof(2) = 0.0;
  end

  if nargin < 2
    order = 2;
  end

  if nargin < 3
    verbose=0;
  end

  if (order == 6)
    stages = 9;
  end
  
  cfl = 0.05;

  Ntot = N+Nguard;

  D = size(pcof,1); # parameter dimension

# handles to time and forcing functions
  if (D==4)
    rfunc = @rf4;
    ifunc = @if4;
    efunc = @ef16;
  elseif (D==5)
    rfunc = @rf5;
    ifunc = @if5;
    efunc = @ef5;
  elseif (D==15)
    rfunc = @rf15;
    ifunc = @if15;
    efunc = @ef15;
  elseif (D==16)
    rfunc = @rf16;
    ifunc = @if16;
    efunc = @ef16;
  elseif (D==20)
    rfunc = @rf20;
    ifunc = @if20;
    efunc = @ef20;
  elseif (D==24)
    rfunc = @rf24;
    ifunc = @if24;
    efunc = @ef24;
  elseif (D==25)
    rfunc = @rf25;
    ifunc = @if25;
    efunc = @ef25;
  elseif (D==30)
    rfunc = @rf30;
    ifunc = @if30;
    efunc = @ef30;
  else
    printf("ERROR: number of parameters D=%d is not implemented\n", D);
    return;
  end
  
  
# coefficients in H0
  omega = zeros(1,Ntot);
  omega(1) = 0;
  omega(2) = 25.798;
  omega(3) = 50.216;
  omega(4) = 73.252;
  if Ntot == 6
    omega(5) = 94.908;
    omega(6) = 115.182;
  end
  lab_frame = 0;
##   if (lab_frame)
## # lab frame
##     H0 = diag([omega(1), omega(2), omega(3), omega(4)]);
##     d_omega = [0, 0, 0, 0];
##   else
# rotating frame
  H0 = zeros(Ntot,Ntot);
  d_omega = zeros(1, Ntot);
  d_omega(1:Ntot-1) = omega(2:Ntot) - omega(1:Ntot-1);
##  end

				# for struct for passing parameters to the time function
  param = struct("pcof", pcof, "T", T, "d_omega", d_omega);

				# lowering op
  if (Ntot==6)
    amat = [0, 1, 0, 0, 0, 0;
  	    0, 0, sqrt(2), 0, 0, 0;
  	    0, 0, 0, sqrt(3), 0, 0;
  	    0, 0, 0, 0, sqrt(4), 0;
  	    0, 0, 0, 0, 0, sqrt(5);
  	    0, 0, 0, 0, 0, 0];
  elseif (Ntot==7)
    amat = [0, 1, 0, 0, 0, 0, 0;
  	    0, 0, sqrt(2), 0, 0, 0, 0;
  	    0, 0, 0, sqrt(3), 0, 0, 0;
  	    0, 0, 0, 0, sqrt(4), 0, 0;
  	    0, 0, 0, 0, 0, sqrt(5), 0;
  	    0, 0, 0, 0, 0, 0, sqrt(6);
  	    0, 0, 0, 0, 0, 0, 0];
  elseif (Ntot==4)
    amat = [0, 1, 0, 0;
  	    0, 0, sqrt(2), 0;
  	    0, 0, 0, sqrt(3);
  	    0, 0, 0, 0];
  else
    printf("ERROR: Ntot = %d is not implemented\n", Ntot);
    objf_v = -999;
    return;
  end
# raising op
  adag = amat';
  
  if (verbose)
    printf("Vector dim (Ntot) = %d, Guard levels (Nguard) = %d, Param dim (D) = %d, pcof(1) = %e, CFL = %e\n",
	   Ntot, Nguard, D, param.pcof(1), cfl);
  end

# rotating frame: time step essentially determined by time scale of forcing
  maxeig1 = max(abs(d_omega))/(2*pi);

# estimate max eigenvalue
  pcofmax = max(abs(pcof));
  K_1 =  pcofmax.*( amat +  amat');
  lambda = eig(K_1);
  maxeig2 = norm(lambda,"inf");
  if (verbose)
    printf("max(d_omega) = %e, maxeig1 = %e, pcofmax = %e, maxeig2 = %e\n", max(abs(d_omega)), maxeig1, ...
	   pcofmax, maxeig2);
  end
  maxeig = 0.5*(maxeig1+ maxeig2);
  
		   # Final time T
  dt = cfl/maxeig; 
  nsteps = ceil(T/dt);
  dt = T/nsteps;
  if (verbose)
    printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   T, nsteps, maxeig, cfl, dt);
  end

# different weight functions for different components
  wconst = 0.01; # for response to e2 and e3
  
  zeroMat = zeros(Ntot,N);
# the basis for the initial data as a matrix
  Ident=diag(ones(1,Ntot));
  U0 = Ident(1:Ntot,1:N);

# Target state at t=T (always real)
  uTarget = Ident(1:Ntot,1:N);
  # CNOT gate by swapping the last and second last column
  uTarget(:,3) = Ident(:,4);
  uTarget(:,4) = Ident(:,3);

  RotMat = diag([ exp(I*omega*T) ]); # Is this syntax correct?

  vTarget = RotMat*uTarget;

# real arithmetic for Verlet
  RotMat_r = diag([ cos(omega*T) ]); # syntax?
  RotMat_i = diag([ sin(omega*T) ]);

# uTarget is real
  vTarget_r = RotMat_r*uTarget;
  vTarget_i = RotMat_i*uTarget;

# initial data and allocation of solution vectors
# real and negative imaginary part of the solution
  ur = U0;
  vi = zeros(Ntot, N);

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
  else
    printf("ERROR: order = %d is not yet implemented\n", order);
    return;
  end
  
# allocate space for saving the time evolution of the solution
  if (verbose)
    usaver = zeros(Ntot,N,nsteps+1);
    usavei = zeros(Ntot,N,nsteps+1);
    usaver(:,:,1) = ur;
    usavei(:,:,1) = -vi;
  end

# for computing the objf function
  objf_v = 0;

# time stepping loop, harmonic oscillator
  t=0;
  step=0;
  for step=1:nsteps

# Stromer-Verlet
    infidelity_0 = weightf(t,T)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega));
    forbidden_0 = xi*penalf(t,T)*norm2_guard(ur, vi, Nguard);

    for q=1:stages
# the following call updates ( t, ur, vr)
      [ur, vi, t] = stromer_verlet_mat3(ur, vi, rfunc, ifunc, t, gamma(q)*dt, param, H0, amat, Ident, d_omega, zeroMat, zeroMat, zeroMat, zeroMat); 
# real arithmetic for Verlet
# accumulate objf = integral w(t) * ( 1 - |Tr( uSol' * vTarget )/N|^2 )
      infidelity = weightf(t,T)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega));
      forbidden = xi*penalf(t,T)*norm2_guard(ur, vi, Nguard);
      objf_v = objf_v + gamma(q)*dt* 0.5* (infidelity_0 + infidelity + forbidden_0 + forbidden);
      infidelity_0 = infidelity;	# save previous values for next stage
      forbidden_0 = forbidden;
    end

# save solutions from both methods to evaluate differences
    if (verbose)
      usaver(:,:,step+1) = ur;
      usavei(:,:,step+1) = -vi;
      
    end # if verbose
  end # for (time stepping loop)

# verlet needs real arithmetic
  RotMat_r = diag([ cos(omega*T) ]);
  RotMat_i = diag([ sin(omega*T)  ]);
  uFinal_r = RotMat_r' *ur - RotMat_i' * vi;
  uFinal_i = -RotMat_r' * vi - RotMat_i * ur;

				# plot results
  if (verbose)
				# difference at final time
    Nplot = nsteps + 1;
				# unitary?
    printf(" Column   Vnrm\n");
    for q=1:N
      Vnrm = usaver(:,q,Nplot)' * usaver(:,q,Nplot) + usavei(:,q,Nplot)' * usavei(:,q,Nplot);
      Vnrm = sqrt(Vnrm);
      printf(" %d  %e\n", q, Vnrm);
    end
			    # tmp: compare solutions from both methods

    tplot = linspace(0,T,Nplot);
    c=3;
    q=3;
    
    plotunitary2(usaver+I*usavei, T, abs_or_real);
    
		# evaluate the polynomials at the discrete time levels
		# evaluate all polynomials on the midpoint grid
    td = linspace(0, T, nsteps+1);
    p_r = rfunc(td,param);
    p_i = ifunc(td,param);
    figure(N+1);
    clf;
    subplot(3,1,1);
    h=plot(td, p_r,"b-");
    legend("Real");
#    h=plot(td, p_r,"b-", td, p_i, "r-");
#    legend("Real",  "Imag");
    axis("tight");
    set(h,"linewidth",2);
    title("Control function");

    subplot(3,1,2);
    envelope = efunc(td, param);
    h = plot(td, envelope, "-");
    axis tight;
    set(h,"linewidth",2);
    Nfreq=size(envelope,1);
    if (Nfreq==4)
      legend("do_1", "do_2", "do_3", "do_4")
    elseif (Nfreq==5)
      legend("do_1", "do_2", "do_3", "do_4", "do_5", "location", "northwest")
    else
      printf("Warning; plotting of envelope function not implemented for Nfreq=%d\n", Nfreq);
    end
    title("Envelope functions");

    subplot(3,1,3);
    wghf1 = weightf(td,T);
    penal1 = penalf(td,T);
    h = plot(td, wghf1, "m", td, penal1, "k--");
    axis tight;
    set(h,"linewidth",2);
    legend("Gate", "Forbidden");
    title("Weight functions");
    
				# output final solution and target
    printf("uTarget:  id1          id2           id3           id4\n");
    for k=1:Ntot
      printf("row=%d: ", k);
      for j=1:N
	printf(" %13.6e", uTarget(k,j));
      end
      printf("\n");
    end

    printf("uFinal:  id1          id2           id3           id4\n");
    for k=1:Ntot
      printf("row=%d: ", k);
      for j=1:N
	printf(" (%13.6e, %13.6e)", uFinal_r(k,j), uFinal_i(k,j) );
      end
      printf("\n");
    end

    uFinal_uTarget = ctranspose(uFinal_r + I*uFinal_i) * uTarget;
    printf("uF' * uTarget:  col1          col2           col3           col4\n");
    for k=1:N
      printf("row=%d: ", k);
      for j=1:N
	printf(" (%13.6e, %13.6e)", real(uFinal_uTarget(k,j)), imag(uFinal_uTarget(k,j)) );
      end
      printf("\n");
    end

    
				# total objf function at final time
    final_fidelity = abs( trace(uFinal_uTarget)/N ); # uTarget is real

    printf("Forward calculation: Parameter pcof =[ %e", param.pcof(1));
    if (D>=2)
      for q=2:D
	printf(", %e", param.pcof(q));
      end
    end
    printf(" ]\n");
				# check if uFinal is unitary
    utest = uFinal_r' * uFinal_r + uFinal_i' * uFinal_i - diag(ones(1,N));
    printf("xi = %e, Final unitary infidelity = %e, Final | trace | gate fidelity = %e\n", xi, norm(utest), final_fidelity);
    printf("Nsteps=%d, Integrated |trace|^2 infidelity = %e\n", nsteps, objf_v);
				# final solution
    infidelity = weightf(T,T)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, T, omega));
    forbidden = xi*penalf(T,T)*norm2_guard(ur, vi, Nguard);
    printf("Last time step: trace2 infidelity = %e, norm2 of guard levels = %e\n", infidelity, forbidden);
				# check guard states
# first sum over the time index, then sum over the columns (1:N)
    abs_sum2 = sum(sumsq(usaver, 3) + sumsq(usavei, 3), 2 )/Nplot/N;
    for q=N+1:N+Nguard
      printf("L2-norm of guard(level=%d) = %e\n", q, sqrt(abs_sum2(q)) );
    end
  end # if verbose
end
