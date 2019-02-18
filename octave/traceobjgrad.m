%-*-octave-*--
%
% % tracegradient: compute the gradient of the trace norm gate infidelity objective functional using the adjoint 
%
% USAGE:
% 
% [objf, grad_objf_adj] = traceobjgrad(pcof0, kpar, dp, order, verbose)
%
% INPUT:
% pcof(D,1): amplitudes of the control functions as a D x 1 column vector, D=size(pcof,1)
% kpar: component of the gradient (1 <= kpar <= D)
% verbose: 0: quite mode, 1: verbose
% order: order of accuracy: 2, 4, or 6.
%
% OUTPUT:
% objf: objective function
% grad_objf_adj: gradient of the objective function computed with an adjoint technique
%
function [objf_v grad_objf_adj ] = traceobjgrad(pcof0, kpar, dp, order, verbose)

  N = 4; # vector dimension
  Nguard = 2;  # number of extra levels
  T=20;# final time
  test_adjoint=0;
  abs_or_real=0; # plot the magnitude (abs) of real part of the solution (1 for real)
  xi=1/N; # coefficient for penalizing forbidden states

  if nargin < 1
    pcof0 = [0.2; 0.1];
  end

  if nargin < 2
    kpar = 1;
  end

  if nargin < 3
    dp = 1e-6;
  end

  if nargin < 4
    order = 2;
  end

  if nargin<5
    verbose=0;
  end

# setup
# for computing the objective function
  objf_v = 0;

  pcof = pcof0;

  if (order == 6)
    stages = 9;
  end
  
  cfl = 0.05;

  Ntot = N+Nguard;

  D = size(pcof,1); # parameter dimension
  grad_objf_adj = zeros(D,1);
  
  if (verbose)
				# first approximate the gradient by FD
    f0 = traceobjf1(pcof0, order);

    pcof1 = pcof0;
    pcof1(kpar) = pcof1(kpar) + dp;

    f1 = traceobjf1(pcof1, order);

				# divided difference approximation
    dfdp_fd = (f1-f0)/dp;

    printf("pcof0: ")
    for q=1:length(pcof0)
      printf(" %e", pcof0(q));
    end
    printf("\n");
    printf("pcof1: ")
    for q=1:length(pcof1)
      printf(" %e", pcof1(q));
    end
    printf("\n");
    printf("dp1 = %e, f1 = %e, f0 = %e\n", dp, f1, f0);
    printf("(f1-f0)/dp = %e\n", dfdp_fd);
  end  
  
# handles to time functions and  gradient of control functions
  if (D==4)
    rfunc = @rf4;
    ifunc = @if4;
    rf_grad = @rf4grad;
    if_grad = @if4grad;
  elseif (D==5)
    rfunc = @rf5;
    ifunc = @if5;
    rf_grad = @rf5grad;
    if_grad = @if5grad;
  elseif (D==15)
    rfunc = @rf15;
    ifunc = @if15;
    rf_grad = @rf15grad;
    if_grad = @if15grad;
    efunc = @ef15;
  elseif (D==16)
    rfunc = @rf16;
    ifunc = @if16;
    rf_grad = @rf16grad;
    if_grad = @if16grad;
  elseif (D==20)
    rfunc = @rf20;
    ifunc = @if20;
    efunc = @ef20;
    rf_grad = @rf20grad;
    if_grad = @if20grad;
  elseif (D==24)
    rfunc = @rf24;
    ifunc = @if24;
    rf_grad = @rf24grad;
    if_grad = @if24grad;
  elseif (D==25)
    rfunc = @rf25;
    ifunc = @if25;
    efunc = @ef25;
    rf_grad = @rf25grad;
    if_grad = @if25grad;
  elseif (D==30)
    rfunc = @rf30;
    ifunc = @if30;
    efunc = @ef30;
    rf_grad = @rf30grad;
    if_grad = @if30grad;
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
# rotating frame
  H0 = zeros(Ntot,Ntot);
  d_omega = zeros(1, Ntot);
  d_omega(1:Ntot-1) = omega(2:Ntot) - omega(1:Ntot-1);

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
# raising op is the transpose of amat
  
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
  dt = cfl/maxeig; # largest eigenvalue of H0 = omega(4), H0+poly*K1 estimated by maxeig
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

  W0 = zeroMat; # initial condition for the phi (d psi/ d alpha1)
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

# initial data for state variable
# real and negative imaginary part
  v_r = U0;
  v_i = zeros(Ntot, N);

# initial data for phi
  w_r = zeroMat;
  w_i = zeroMat;
  
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
  
  if (verbose)
    usaver = zeros(Ntot,N,nsteps+1);
    usavei = zeros(Ntot,N,nsteps+1);
    usaver(:,:,1) = v_r;
    usavei(:,:,1) = -v_i;
  end
  
# for computing the objf_alpha1 function
  objf_alpha1 = 0;

  t=0;
  step=0;
			     # Forward time stepping loop
  for step=1:nsteps

    if (test_adjoint)
      s_cmplx_0 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);
      s_alpha_0 = trace2_fid_cmplx(w_r, -w_i, vTarget_r, vTarget_i, t, omega);
      forb_alpha_0 = xi*penalf(t,T)*sc_real(v_r, v_i, w_r, w_i, Nguard);

# forcing for evolving W (d psi/d alpha1) in the rotating frame
      [da_r, da_i] = get_da_mat(t, amat, d_omega);
      rgrad = rf_grad(t,param);
      igrad = if_grad(t,param);
      rf_alpha = rgrad(kpar);
      if_alpha = igrad(kpar);
      gr_0 = rf_alpha.*( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i ) + if_alpha.*(  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
      gi_0 = rf_alpha.*( (da_r + da_r') * v_r + (da_i -  da_i') * v_i ) + if_alpha.*( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
    end # test_adjoint

# for the objective function
    infidelity_0 = weightf(t,T)*(1-trace_fid_real(v_r, v_i, vTarget_r, vTarget_i, lab_frame, t, omega));
    forbidden_0 = xi*penalf(t,T)*norm2_guard(v_r, v_i, Nguard);
    
# Stromer-Verlet
    for q=1:stages
      t0=t;
      v_r0 = v_r;
      v_i0 = v_i;
# the following call updates ( t, v_r, v_i)
      [v_r, v_i, t] = stromer_verlet_mat3(v_r, v_i, rfunc, ifunc, t, gamma(q)*dt, param, H0, amat, Ident, d_omega, zeroMat, zeroMat, zeroMat, zeroMat); 

# accumulate objf = integral w(t) * ( 1 - |Tr( uSol' * vTarget )/N|^2 )
      infidelity = weightf(t,T)*(1-trace_fid_real(v_r, v_i, vTarget_r, vTarget_i, lab_frame, t, omega));
      forbidden = xi*penalf(t,T)*norm2_guard(v_r, v_i, Nguard);
      objf_v = objf_v + gamma(q)*dt* 0.5* (infidelity_0 + infidelity + forbidden_0 + forbidden);
# save previous values for next stage
      infidelity_0 = infidelity;	
      forbidden_0 = forbidden;
      
      if (test_adjoint)
# real arithmetic for Verlet
	s_cmplx_1 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);

# forcing for evolving W (d psi/d alpha1) in the rotating frame
	[da_r, da_i] = get_da_mat(t, amat, d_omega);
	rgrad = rf_grad(t,param);
	igrad = if_grad(t,param);
	rf_alpha = rgrad(kpar);
	if_alpha = igrad(kpar);

	gr_1 = rf_alpha.*( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i ) + if_alpha.*(  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
	gi_1 = rf_alpha.*( (da_r + da_r') * v_r + (da_i -  da_i') * v_i ) + if_alpha.*( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
# evolve ( w_r, w_i)
	[w_r, w_i] = stromer_verlet_mat3(w_r, w_i, rfunc, ifunc, t0, gamma(q)*dt, param, H0, amat, Ident, d_omega, gr_0, gr_1, gi_0, gi_1); 
	s_alpha_1 = trace2_fid_cmplx(w_r, -w_i, vTarget_r, vTarget_i, t, omega);
	forb_alpha_1 = xi*penalf(t,T)*sc_real(v_r, v_i, w_r, w_i, Nguard);
      
# accumulate integrated sensitivity
	objf_alpha1 = objf_alpha1 - gamma(q)*dt* 0.5* 2.0 * real( weightf(t0,T) * conj(s_cmplx_0) * s_alpha_0 +  weightf(t,T) * conj(s_cmplx_1) * s_alpha_1) +  gamma(q)*dt* 0.5* 2.0 * (forb_alpha_0 + forb_alpha_1);

# save previous values for next stage
	s_cmplx_0 = s_cmplx_1;
	s_alpha_0 = s_alpha_1;
	forb_alpha_0 = forb_alpha_1;
	gr_0 = gr_1;
	gi_0 = gi_1;
      end  #test_adjoint
    end

# save the solution from the forwards time stepping for plotting and checking the time-reversed calculation
    if (verbose)
      usaver(:,:,step+1) = v_r;
      usavei(:,:,step+1) = -v_i;
      
    end # if verbose
  end # for (time stepping loop)

  dfdp = objf_alpha1;

# evaluate final solution in the lab frame
# verlet needs real arithmetic
  RotMat_r = diag([ cos(omega*T) ]);
  RotMat_i = diag([ sin(omega*T)  ]);
  uFinal_r = RotMat_r' * v_r - RotMat_i' * v_i;
  uFinal_i = -RotMat_r' * v_i - RotMat_i * v_r;

# reverse time step the state variable and the adjoint wave equation
  # initial condition for the state variable (psi) from final solution (vr, vi)
  t=T;
  dt = -dt;
  adiff_max = 0;
  
# terminal conditions for the adjoint state
  lambda_r = zeroMat;
  lambda_i = zeroMat;
  
# reverse time stepping loop
  for step=nsteps-1:-1:0

# Stromer-Verlet
    s_cmplx_0 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);
    sr_0 = real(s_cmplx_0);
    si_0 = imag(s_cmplx_0);
#    hmat_0 = - weightf(t,T) * (sr_0 - I*si_0) * (vTarget_r + I * vTarget_i);
    hr_0 =  -weightf(t,T)/N * (sr_0 * vTarget_r + si_0 *vTarget_i);
    hi_0 =  weightf(t,T)/N * (sr_0 * vTarget_i - si_0 *vTarget_r);
# forcing for guard states
    hr_0(N+1:N+Nguard,:) = xi*penalf(t,T)*v_r(N+1:N+Nguard,:);
    hi_0(N+1:N+Nguard,:) = xi*penalf(t,T)*v_i(N+1:N+Nguard,:);
# forcing for evolving W (d psi/d alpha1) in the rotating frame
    [da_r, da_i] = get_da_mat(t, amat, d_omega);

# separate out contributions from rf_grad and if_grad (which determine the component of the gradient)
    dar_r = ( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i );
    dar_i = ( (da_r + da_r') * v_r + (da_i -  da_i') * v_i );
    dai_r = (  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
    dai_i = ( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
    tr_adj_rf = trace2_fid_real(dar_r, dar_i, lambda_r, lambda_i);
    tr_adj_if = trace2_fid_real(dai_r,  dai_i, lambda_r, lambda_i);
    tr_adj_0 = rf_grad(t, param) * tr_adj_rf + if_grad(t, param) * tr_adj_if;

# loop over the stages
    for q=1:stages
      t0=t;
      v_r0 = v_r;
      v_i0 = v_i;
# the following call updates ( t, v_r, v_i)
      [v_r, v_i, t] = stromer_verlet_mat3(v_r, v_i, rfunc, ifunc, t, gamma(q)*dt, param, H0, amat, Ident, d_omega, zeroMat, zeroMat, zeroMat, zeroMat); 
# real arithmetic for Verlet
      s_cmplx_1 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);
      sr_1 = real(s_cmplx_1);
      si_1 = imag(s_cmplx_1);
				# forcing for the adjoint equation
      hr_1 =  -weightf(t,T)/N * (sr_1 * vTarget_r + si_1 *vTarget_i);
      hi_1 =  weightf(t,T)/N * (sr_1 * vTarget_i - si_1 *vTarget_r);
# forcing for guard states (note that the last Nguard rows of vTarget = 0)
      hr_1(N+1:N+Nguard,:) = xi*penalf(t,T)*v_r(N+1:N+Nguard,:);
      hi_1(N+1:N+Nguard,:) = xi*penalf(t,T)*v_i(N+1:N+Nguard,:);

# evolve lambda_r, lambda_i
      [lambda_r, lambda_i] = stromer_verlet_mat3(lambda_r, lambda_i, rfunc, ifunc, t0, gamma(q)*dt, param, H0, amat, Ident, d_omega, hr_0, hr_1, hi_0, hi_1); 

# forcing for evolving W (d psi/d alpha1) in the rotating frame
      [da_r, da_i] = get_da_mat(t, amat, d_omega);

# separate out contributions from rf_grad and if_grad (which determine the component of the gradient)
      dar_r = ( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i );
      dar_i = ( (da_r + da_r') * v_r + (da_i -  da_i') * v_i );
      dai_r = (  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
      dai_i = ( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
      tr_adj_rf = trace2_fid_real(dar_r, dar_i, lambda_r, lambda_i);
      tr_adj_if = trace2_fid_real(dai_r,  dai_i, lambda_r, lambda_i);
      tr_adj_1 = rf_grad(t, param) * tr_adj_rf + if_grad(t, param) * tr_adj_if;
		 # accumulate the gradient of the objective functional
      grad_objf_adj = grad_objf_adj + gamma(q)*dt* 0.5* 2.0 * ( tr_adj_0 +  tr_adj_1); # dt is negative
      
				# save previous values for next stage
      s_cmplx_0 = s_cmplx_1;
      tr_adj_0 = tr_adj_1;
      hr_0 = hr_1;
      hi_0 = hi_1;
    end

# evaluate differences with forward time-stepping
    if (verbose)
      rdiff = norm(usaver(:,:,step+1) - v_r);
      idiff = norm(usavei(:,:,step+1) + v_i);
      adiff = sqrt(rdiff^2 + idiff^2);
      if (adiff> adiff_max)
	adiff_max = adiff;
      end
    end # if verbose

  end # for (reverse time stepping loop)

  if verbose
    printf("Max diff forward - reverse time stepping: %e\n", adiff_max);
  end
  
				# plot results
  if (verbose)
				# difference at final time
    Nplot = nsteps + 1;
				# unitary?
    printf(" Column   Vnrm\n");
    for q=1:N
      Vnrm = usaver(:,q,Nplot)' * usaver(:,q,Nplot) + usavei(:,q,Nplot)' * usavei(:,q,Nplot);
      Vnrm = sqrt(Vnrm);
##      Mnrm = norm(usave(:,q,Nplot));
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
#    envelope = efunc(td, param);
    envelope = efunc(td, param);
    h = plot(td, envelope, "-");
    axis tight;
    set(h,"linewidth",2);
    Nfreq=size(envelope,1);
    if (Nfreq==4)
      legend("d-om_1", "d-om_2", "d-om_3", "d-om_4")
    elseif (Nfreq==5)
      legend("d-om_1", "d-om_2", "d-om_3", "d-om_4", "d-om_5")
    else
      printf("Warning; plotting of envelope function not implemented for Nfreq=%d\n", Nfreq);
    end
    title("Envelope functions");

    subplot(3,1,3);
    wghf1 = weightf(td,T);
    penal1 = penalf(td,T);
    h = plot(td, wghf1, "m", td, penal1, "k");
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
    for q=2:D
      printf(", %e", param.pcof(q));
    end
    printf(" ]\n");
				# check if uFinal is unitary
    utest = uFinal_r' * uFinal_r + uFinal_i' * uFinal_i - diag(ones(1,N));
    printf("xi = %e, objf = %e, Final unitary infidelity = %e, Final | trace | gate fidelity = %e\n", xi, objf_v, norm(utest), final_fidelity);
    printf("Nsteps=%d, kpar = %d, fd-gradient of objective function = %e\n", nsteps, kpar, dfdp_fd)
    if (test_adjoint) printf("Forward integration of gradient of objective function = %e\n", dfdp);
    printf("Adjoint gradient components: ");
    for q=1:D
      printf(" %e ", grad_objf_adj(q) );
    end
    printf("\n");
  end # if verbose
end
