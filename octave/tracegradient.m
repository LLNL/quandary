%-*-octave-*--
%
% % tracegradient: compute the gradient of the trace norm gate infidelity objective functional
%
% USAGE:
% 
% [dfdp, dfdp_fd] = tracegradient(pcof0, kpar, dp, order, verbose)
%
% INPUT:
% pcof(D,1): amplitudes of the control functions as a D x 1 column vector, D=size(pcof,1)
% kpar: component of the gradient (1 <= kpar <= D)
% verbose: 0: quite mode, 1: verbose
% order: order of accuracy: 2, 4, or 6.
%
% OUTPUT:
% grad_objf_adj: gradient of trace norm objective functional computed with an adjoint technique
%
function [ grad_objf_adj ] = tracegradient(pcof0, kpar, dp, order, verbose)

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

  test_adjoint=0;

  if test_adjoint
    if (kpar == 1)
      rfunc_a1 = @rf1alpha1;
      ifunc_a1 = @if1alpha1;
    elseif (kpar == 2)
      rfunc_a1 = @rf1alpha2;
      ifunc_a1 = @if1alpha2;
    else
      printf("ERROR: kpar = %d is not yet implemented\n", kpar);
      return;
    end
  end


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

# Next solve the ODE for psi and phi = d psi/d alpha
  abs_or_real=1; # plot the magnitude (abs) of real part of the solution (1 for real)

  pcof = pcof0;

  if (order == 6)
    stages = 9;
  end
  
  cfl = 0.1;

  N = 4; # vector dimension

  D = size(pcof,1); # parameter dimension
  
# handles to time functions
  if (D==2)
    rfunc = @rf1;
    ifunc = @if1;
# gradient of control functions
    rf_grad = @rf1grad;
    if_grad = @if1grad;
  elseif (D==6)
    rfunc = @rf6;
    ifunc = @if6;
    rf_grad = @rf6grad;
    if_grad = @if6grad;
  elseif (D==8)
    rfunc = @rf8;
    ifunc = @if8;
    rf_grad = @rf8grad;
    if_grad = @if8grad;
  elseif (D==24)
    rfunc = @rf24;
    ifunc = @if24;
    rf_grad = @rf24grad;
    if_grad = @if24grad;
  else
    printf("ERROR: number of parameters D=%d is not implemented\n", D);
    return;
  end

# coefficients in H0
  omega = zeros(1,4);
  omega(1) = 0;
  omega(2) = 24.64579437;
  omega(3) = 47.88054868;
  omega(4) = 69.70426293;

  lab_frame = 0;
# rotating frame
  H0 = diag([0, 0, 0, 0]);
  d_omega = [omega(2)-omega(1), omega(3)-omega(2), omega(4)-omega(3), 0];

# lowering op
  amat = [0, 1, 0, 0;
  	0, 0, sqrt(2), 0;
  	0, 0, 0, sqrt(3);
  	0, 0, 0, 0];
# raising op is the transpose of amat
  
# final time
  global T;

  if (verbose)
    printf("Vector dim (N) = %d, Param dim (D) = %d, pcof(1) = %e, Final time = %e, CFL = %e\n", N, D, pcof(1), T, cfl);
  end

# rotating frame: time step essentially determined by time scale of forcing
  maxeig = max(abs(d_omega))/(2*pi);

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
  
  zeroMat = zeros(N,N);
# the basis for the initial data as a matrix
  Ident=diag([1, 1, 1, 1]);
  U0 = Ident; # initial condition for the state variable (psi)
  W0 = zeroMat; # initial condition for the phi (d psi/ d alpha1)

# Target state at t=T (always real)
  uTarget = [0, 1, 0, 0;
	     1, 0, 0, 0;
	     0, 0, 1, 0;
	     0, 0, 0, 1];
  
  RotMat = diag([ exp(I*omega(1)*T), exp(I*omega(2)*T), exp(I*omega(3)*T), exp(I*omega(4)*T) ]);
  vTarget = RotMat*uTarget;
				# real arithmetic for Verlet
  RotMat_r = diag([ cos(omega(1)*T), cos(omega(2)*T), cos(omega(3)*T), cos(omega(4)*T) ]);
  RotMat_i = diag([ sin(omega(1)*T), sin(omega(2)*T), sin(omega(3)*T), sin(omega(4)*T) ]);
# uTarget is real
  vTarget_r = RotMat_r*uTarget;
  vTarget_i = RotMat_i*uTarget;

# initial data for state variable
# real and negative imaginary part
  v_r = U0;
  v_i = zeroMat;

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
    printf("ERROR: order = %d is not yet implemented\n");
    return;
  end
  
  if (verbose)
    usaver = zeros(N,N,nsteps+1);
    usavei = zeros(N,N,nsteps+1);
    usaver(:,:,1) = v_r;
    usavei(:,:,1) = -v_i;
  end
  
  separable = 0;

# for computing the objf_alpha1 function
  objf_alpha1 = 0;

  t=0;
  step=0;
			     # Forward time stepping loop
  for step=1:nsteps

# Stromer-Verlet
    if (test_adjoint)
      s_cmplx_0 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);
      s_alpha_0 = trace2_fid_cmplx(w_r, -w_i, vTarget_r, vTarget_i, t, omega);

# forcing for evolving W (d psi/d alpha1) in the rotating frame
      [da_r, da_i] = get_da_mat(t, amat, d_omega);
      rf_alpha = rfunc_a1(t,pcof);
      if_alpha = ifunc_a1(t,pcof);
      gr_0 = rf_alpha.*( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i ) + if_alpha.*(  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
      gi_0 = rf_alpha.*( (da_r + da_r') * v_r + (da_i -  da_i') * v_i ) + if_alpha.*( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
    end # test_adjoint
    
    for q=1:stages
      t0=t;
      v_r0 = v_r;
      v_i0 = v_i;
# the following call updates ( t, v_r, v_i)
      [v_r, v_i, t] = stromer_verlet_mat2(v_r, v_i, rfunc, ifunc, t, gamma(q)*dt, pcof, H0, amat, Ident, d_omega, zeroMat, zeroMat, zeroMat, zeroMat); 

      if (test_adjoint)
# real arithmetic for Verlet
	s_cmplx_1 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);

# forcing for evolving W (d psi/d alpha1) in the rotating frame
	[da_r, da_i] = get_da_mat(t, amat, d_omega);
	rf_alpha = rfunc_a1(t,pcof);
	if_alpha = ifunc_a1(t,pcof);

	gr_1 = rf_alpha.*( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i ) + if_alpha.*(  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
	gi_1 = rf_alpha.*( (da_r + da_r') * v_r + (da_i -  da_i') * v_i ) + if_alpha.*( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
# evolve ( w_r, w_i)
	[w_r, w_i] = stromer_verlet_mat2(w_r, w_i, rfunc, ifunc, t0, gamma(q)*dt, pcof, H0, amat, Ident, d_omega, gr_0, gr_1, gi_0, gi_1); 

	s_alpha_1 = trace2_fid_cmplx(w_r, -w_i, vTarget_r, vTarget_i, t, omega);
      
# accumulate integrated sensitivity
	objf_alpha1 = objf_alpha1 - gamma(q)*dt* 0.5* 2.0 * real( weightf(t0) * conj(s_cmplx_0) * s_alpha_0 +  weightf(t) * conj(s_cmplx_1) * s_alpha_1);

# save previous values for next stage
	s_cmplx_0 = s_cmplx_1;
	s_alpha_0 = s_alpha_1;
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
  RotMat_r = diag([ cos(omega(1)*T), cos(omega(2)*T), cos(omega(3)*T), cos(omega(4)*T) ]);
  RotMat_i = diag([ sin(omega(1)*T), sin(omega(2)*T), sin(omega(3)*T), sin(omega(4)*T) ]);
  uFinal_r = RotMat_r' * v_r - RotMat_i' * v_i;
  uFinal_i = -RotMat_r' * v_i - RotMat_i * v_r;

# reverse time step the state variable and the adjoint wave equation
  # initial condition for the state variable (psi) from final solution (vr, vi)
  t=T;
  dt = -dt;
  adiff_max = 0;
  grad_objf_adj = zeros(D,1);
  
# terminal conditions for the adjoint state
  lambda_r = zeroMat;
  lambda_i = zeroMat;
  
# reverse time stepping loop
  for step=nsteps-1:-1:0

# Stromer-Verlet
    s_cmplx_0 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);
    sr_0 = real(s_cmplx_0);
    si_0 = imag(s_cmplx_0);
#    hmat_0 = - weightf(t) * (sr_0 - I*si_0) * (vTarget_r + I * vTarget_i);
    hr_0 =  -weightf(t) * (sr_0 * vTarget_r + si_0 *vTarget_i);
    hi_0 =  weightf(t) * (sr_0 * vTarget_i - si_0 *vTarget_r);

# forcing for evolving W (d psi/d alpha1) in the rotating frame
    [da_r, da_i] = get_da_mat(t, amat, d_omega);

# separate out contributions from rf_grad and if_grad (which determine the component of the gradient)
      dar_r = ( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i );
      dar_i = ( (da_r + da_r') * v_r + (da_i -  da_i') * v_i );
      dai_r = (  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
      dai_i = ( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
      tr_adj_rf = trace2_fid_real(dar_r, dar_i, lambda_r, lambda_i, t, omega);
      tr_adj_if = trace2_fid_real(dai_r,  dai_i, lambda_r, lambda_i, t, omega);
      tr_adj_0 = rf_grad(t, pcof) * tr_adj_rf + if_grad(t, pcof) * tr_adj_if;

# loop over the stages
    for q=1:stages
      t0=t;
      v_r0 = v_r;
      v_i0 = v_i;
# the following call updates ( t, v_r, v_i)
      [v_r, v_i, t] = stromer_verlet_mat2(v_r, v_i, rfunc, ifunc, t, gamma(q)*dt, pcof, H0, amat, Ident, d_omega, zeroMat, zeroMat, zeroMat, zeroMat); 
# real arithmetic for Verlet
      s_cmplx_1 = trace2_fid_cmplx(v_r, -v_i, vTarget_r, vTarget_i, t, omega);
      sr_1 = real(s_cmplx_1);
      si_1 = imag(s_cmplx_1);
				# forcing for the adjoint equation
      hr_1 =  -weightf(t) * (sr_1 * vTarget_r + si_1 *vTarget_i);
      hi_1 =  weightf(t) * (sr_1 * vTarget_i - si_1 *vTarget_r);

# evolve lambda_r, lambda_i
      [lambda_r, lambda_i] = stromer_verlet_mat2(lambda_r, lambda_i, rfunc, ifunc, t0, gamma(q)*dt, pcof, H0, amat, Ident, d_omega, hr_0, hr_1, hi_0, hi_1); 

# forcing for evolving W (d psi/d alpha1) in the rotating frame
      [da_r, da_i] = get_da_mat(t, amat, d_omega);

# separate out contributions from rf_grad and if_grad (which determine the component of the gradient)
      dar_r = ( (da_i -  da_i') * v_r  - (da_r + da_r') * v_i );
      dar_i = ( (da_r + da_r') * v_r + (da_i -  da_i') * v_i );
      dai_r = (  (da_i + da_i') * v_i + (da_r  -  da_r') * v_r );
      dai_i = ( -(da_i + da_i') * v_r + (da_r  -  da_r') * v_i );
      tr_adj_rf = trace2_fid_real(dar_r, dar_i, lambda_r, lambda_i, t, omega);
      tr_adj_if = trace2_fid_real(dai_r,  dai_i, lambda_r, lambda_i, t, omega);
      tr_adj_1 = rf_grad(t, pcof) * tr_adj_rf + if_grad(t, pcof) * tr_adj_if;
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
    printf(" Initial data   Vnrm\n");
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
    
    plotunitary(usaver+I*usavei, T, abs_or_real);
    
		# evaluate the polynomials at the discrete time levels
		# evaluate all polynomials on the midpoint grid
    td = linspace(0, T, nsteps+1);
    p_r = rfunc(td,pcof);
    p_i = ifunc(td,pcof);
    figure(5);
    subplot(2,1,1);
    h=plot(td, p_r,"b", td, p_i, "r");
    legend("Real",  "Imag");
    axis("tight");
    set(h,"linewidth",2);
    title("Forcing function");

    subplot(2,1,2);
    wghf1 = weightf(td);
    h = plot(td, wghf1, "m");
    axis tight;
    set(h,"linewidth",2);
    title("Weight function");

				# output final solution and target
    printf("uTarget:  id1          id2           id3           id4\n");
    for k=1:N
      printf("k=%d: ", k);
      for j=1:N
	printf(" %13.6e", uTarget(j,k));
      end
      printf("\n");
    end

    printf("uSol-Ve:  id1          id2           id3           id4\n");
    for k=1:N
      printf("k=%d: ", k);
      for j=1:N
	printf(" %13.6e", abs(uFinal_r(j,k) + I*uFinal_i(j,k))  );
      end
      printf("\n");
    end
				# total objf function at final time
    final_Infidelity = 1 - abs(trace(uFinal_r' * uTarget)/N); # uTarget is real

    printf("Forward calculation: Parameter pcof =[ %e", pcof(1));
    for q=2:D
      printf(", %e", pcof(q));
    end
    printf(" ]\n");
				# check if uFinal is unitary
    utest = uFinal_r' * uFinal_r + uFinal_i' * uFinal_i - U0;
    printf("LabFrame = %d, Final unitary infidelity = %e, Final | trace | gate infidelity = %e\n", lab_frame, norm(utest), final_Infidelity);
    printf("Nsteps=%d, kpar = %d, fd-gradient of objective function = %e\n", nsteps, kpar, dfdp_fd)
    printf("Adjoint gradient components: ");
    for q=1:D
      printf(" %e ", grad_objf_adj(q) );
    end
    printf("\n");
  end # if verbose
end
