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
function [objf_v] = traceobjf1(pcof, order, verbose)

  N = 4; # vector dimension
  Nguard = 3; # number of extra levels
  T=150;# final time
  xi=1.0/Nguard; # coefficient for penalizing forbidden states
  abs_or_real=0; # plot the magnitude (abs) of real part of the solution (1 for real)
  par_1 = 0.09; # max value of parameters
  par_0 = -par_1;
  eps = 1e-9;

  if nargin < 1
    pcof(1) = 0.0;
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

	 # verify that each element of pcof is in the prescribed range
  for q=1:D
    if (pcof(q) > par_1-eps)
      pcof(q) = par_1-eps;
    end
    if (pcof(q) < par_0+eps)
      pcof(q) = par_0+eps;
    end
  end
  
  
# handles to time and forcing functions
  ## nurbs_control = 0;
  ## if (D==4)
  ##   rfunc = @rf4;
  ##   ifunc = @if4;
  ##   efunc = @ef16;
  ## elseif (D==5)
  ##   rfunc = @rf5;
  ##   ifunc = @if5;
  ##   efunc = @ef5;
  ## elseif (D==15)
  ##   rfunc = @rf15;
  ##   ifunc = @if15;
  ##   efunc = @ef15;
  ## elseif (D==16)
  ##   rfunc = @rf16;
  ##   ifunc = @if16;
  ##   efunc = @ef16;
  ## elseif (D==20)
  ##   rfunc = @rf20;
  ##   ifunc = @if20;
  ##   efunc = @ef20;
  ## else
    if (verbose)
      printf("Assuming a Nurbs parameterization with %d nurbs wavelets\n", D);
    end
    nurbs_control = 1;
    rfunc = @nurb2;
    ifunc = @zero_func;
    efunc = @nurb2;
    ##  end
  
  
# coefficients in H0
  omega = zeros(1,Ntot);
  omega(1) = 0;
  omega(2) = 4.106;
  omega(3) = 7.992;
  omega(4) = 11.659;
  if Ntot >= 6
    omega(5) = 15.105;
    omega(6) = 18.332;
  end
  if Ntot >= 7
    omega(7) = 21.339;
  end
  if Ntot > 7
    printf("ERROR: Not enough frequencies are known for Ntot=%d\n", Ntot);
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
  if (nurbs_control)
    N_nurbs = D;
    N_knots = N_nurbs+1;
    dt_knot = T/(N_nurbs-2);
    width = 3*dt_knot;
    t_center = dt_knot*([1:N_nurbs] - 1.5);
    t_knot = dt_knot*( [1:N_knots] - 2 );

    ## for q=1:2:N_nurbs
    ##   pcof(q) = -1;
    ## end
    ## pcof = 2*rand(N_nurbs,1) - 1;
    ## pcof(1:2) = 0;
    ## pcof(N_nurbs) = 0;
    ## pcof(N_nurbs - 1) = 0;
    
 # param: struct for passing parameters to the time function
 # T is a real positive scalar
 # N_nurbs: number of nurbs ( positive integer ) 
 # N_knots: number of knots (N_nurbs+3)
 # t_center: Center time for each nurb (1 x N_nurbs) array of real
 # t_knot: Time corresponding to each knot (1 x N_knots) array of real
 # dt_knot: Spacing in t_knot array
 # pcof: Nurbs coefficients (N_nurbs x 1) array of real
    param = struct("T", T, "N_nurbs", N_nurbs, "t_knot", t_knot, "dt_knot", dt_knot, "t_center", t_center, "pcof", pcof);
  else
    param = struct("pcof", pcof, "T", T, "d_omega", d_omega);
  end
  
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

			       # Inequality constraints on parameters:
  ineq_penalty =  eval_ineq_pen(pcof, par_0, par_1);
  objf_v = objf_v + ineq_penalty;
  
				# plot results
  if (verbose)
    printf("Inequality penalty = %e\n", ineq_penalty);
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
      for q=2:min(10,D)
	printf(", %e", param.pcof(q));
      end
    end
    if (D > 10)
      printf(" (truncated to 10 elements) ");
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

# FFT on the control function
    tdf = linspace(dt, T, nsteps);
    ctrl = rfunc(tdf,param);

    df = 1/T;
    Nf = nsteps;
    freq=[ -(ceil((Nf-1)/2):-1:1), 0, (1:floor((Nf-1)/2)) ] * df; # In Hz
    om = 2*pi*freq; % angular frequency in rad/sec

    fctrl = fftshift(fft(ifftshift(ctrl)));
    fctrl = fctrl/Nf;

    figure(N+2);
    semilogy(om, abs(fctrl) + 1e-18, "r+");  # 1e-18 is to avoid warning messages
    axis([0, 20, 1e-5, max(abs(fctrl))]);
    npar = length(pcof);
    tstr = sprintf("FFT(drive), %d B-spline wavelets", D);
    title(tstr);
    xlabel("Omega [rad/s]");
  end # if verbose
end
