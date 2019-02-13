%-*-octave-*--
%
% traceobjfunc: solve a model problem from quantum control theory
%
% USAGE:
% 
% [objF, uFinal] = traceobjfunc(pcof, verbose, order)
%
% INPUT:
% pcof(D,1): amplitudes of the control functions as a D x 1 column vector, D=size(pcof,1)
% verbose: 0: quite mode, 1: verbose
% order: order of accuracy: 2, 4, or 6.
%
% OUTPUT:
% objF: trace norm of gate infidelity cost functional
% uFinal: state vector at t=T
%
function [objf_v uFinal] = traceobjfunc(pcof, order, verbose)

  abs_or_real=0; # plot the magnitude (abs) of real part of the solution (1 for real)

  if nargin < 1
    pcof(1) = 1.0;
    pcof(2) = 0.0;
  end

  if nargin < 3
    verbose=0;
  end

  if nargin < 2
    order = 2;
  end

  if (order == 6)
    stages = 9;
  end
  
  cfl = 0.1;

  N = 4; # vector dimension
  Nguard = 2;
  Ntot = N+Nguard;

  D = size(pcof,1); # parameter dimension
  
# handles to time and forcing functions
  if (D==2)
    rfunc = @rf1;
    ifunc = @if1;
  elseif (D==6)
    rfunc = @rf6;
    ifunc = @if6;
  elseif (D==8)
    rfunc = @rf8;
    ifunc = @if8;
  elseif (D==12)
    rfunc = @rf12;
    ifunc = @if12;
  elseif (D==18)
    rfunc = @rf18;
    ifunc = @if18;
  elseif (D==24)
    rfunc = @rf24;
    ifunc = @if24;
  else
    printf("ERROR: number of parameters D=%d is not implemented\n", D);
    return;
  end

# coefficients in H0
  omega = zeros(1,Ntot);
  ## omega(1) = 0;
  ## omega(2) = 24.64579437;
  ## omega(3) = 47.88054868;
  ## omega(4) = 69.70426293;
  omega(1) = 0;
  omega(2) = 25.798;
  omega(3) = 50.216;
  omega(4) = 73.252;
  omega(5) = 94.908;
  omega(6) = 115.182;

  lab_frame = 0;
  if (lab_frame)
# lab frame
    H0 = diag([omega]);
    d_omega = zeros(1, Ntot);
  else
# rotating frame
    H0 = zeros(Ntot, Ntot);
    d_omega = zeros(1, Ntot);
    d_omega(1:Ntot-1) = omega(2:Ntot) - omega(1:Ntot-1);
  end

				# lowering op
  amat = [0, 1, 0, 0, 0, 0;
  	0, 0, sqrt(2), 0, 0, 0;
  	0, 0, 0, sqrt(3), 0, 0;
  	0, 0, 0, 0, sqrt(4), 0;
  	0, 0, 0, 0, 0, sqrt(5);
  	0, 0, 0, 0, 0, 0];
# raising op
  adag = amat';
  
# final time
  global T;

  if (verbose)
    printf("Vector dim (Ntot) = %d, Guard levels (Nguard) = %d, Param dim (D) = %d, pcof(1) = %e, CFL = %e\n",
	   Ntot, Nguard, D, pcof(1), cfl);
  end

  if (lab_frame) # max eigenvalue determined by H0 + max(control terms)
    H=H0+pcof(1)*(amat+amat'); #+I*pcof(2)*(amat-adag);
				# estimate largest eigenvalue
    lambda = eig(H);
    maxeig1 = norm(lambda,"inf");
    if (verbose)
      printf("maxeig1 = %e\n", maxeig1);
    end

    H=H0-pcof(1)*(amat+amat'); #-I*pcof(2)*(amat-adag);
    lambda = eig(H);
    maxeig2 = norm(lambda,"inf");
    if (verbose)
      printf("maxeig2 = %e\n", maxeig2);
    end
    maxeig = max(maxeig1, maxeig2);
  else
# rotating frame: time step essentially determined by fastest time scale in forcing
    maxeig = max(abs(d_omega))/(2*pi);
  end

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

# Target state at t=T (always real), only applies to the first N components of the wave functions
  uTarget = Ident(1:Ntot,1:N);
# CNOT gate by swapping the last and second last column
  uTarget(:,N-1) = Ident(:,N);
  uTarget(:,N) = Ident(:,N-1);
  
  RotMat = diag([ exp(I*omega*T) ]); # Is this syntax correct?
  vTarget = RotMat*uTarget;
				# real arithmetic for Verlet
  RotMat_r = diag([ cos(omega*T) ]); # syntax?
  RotMat_i = diag([ sin(omega*T) ]);
# uTarget is real
  vTarget_r = RotMat_r * uTarget;
  vTarget_i = RotMat_i * uTarget;

# initial data and allocation of solution vectors
  U0 = Ident(1:Ntot,1:N); # only N columns
  uSol = U0;
# real and negative imaginary part of the solution
  ur = U0;
  vi = zeros(Ntot, N);

# setup the time integrator
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
  end
  
# allocate space for saving the time evolution of the solution
  if (verbose)
    usave = zeros(Ntot,N,nsteps+1);
    usaver = zeros(Ntot,N,nsteps+1);
    usavei = zeros(Ntot,N,nsteps+1);
    usave(:,:,1) = uSol;
    usaver(:,:,1) = ur;
    usavei(:,:,1) = -vi;
  end
  
# for computing the objf function
  objf = 0;
  objf_v = 0;

# time stepping loop, harmonic oscillator
  t=0;
  tm=0;
  step=0;
  for step=1:nsteps
# 2nd order Magnus integrator
#    dmat = expm(-I*diag(d_omega.*(tm+0.5*dt)));
    dmat_r_1o2 = diag([ cos(d_omega*(tm+0.5*dt)) ]);
    dmat_i_1o2 = diag([ -sin(d_omega*(tm+0.5*dt)) ]);

				# symmetric part
    K_1o2 =  rfunc(tm+0.5*dt, pcof).*(dmat_r_1o2 * amat +  amat' * dmat_r_1o2') - ifunc(tm+0.5*dt, pcof).*(dmat_i_1o2 * amat + amat' * dmat_i_1o2');
				# skew-symmetric part
    S_1o2 =  ifunc(tm+0.5*dt, pcof).*(dmat_r_1o2 * amat - amat' * dmat_r_1o2') + rfunc(tm+0.5*dt, pcof).*(dmat_i_1o2 * amat - amat' * dmat_i_1o2');
			       
    ## H = H0 + (rfunc(tm+0.5*dt, pcof) + I*ifunc(tm+0.5*dt, pcof)).*(da_mat_r + I*da_mat_i) + ...
    ## 	(rfunc(tm+0.5*dt, pcof) - I*ifunc(tm+0.5*dt, pcof)).*(da_mat_r' - I*da_mat_i'); # symmetric + skew-symmtric 

    H = H0 + K_1o2 + I*S_1o2; # symmetric + skew-symmtric 
    expH = expm(-I*dt*H);
    uSol = expH * uSol;
    tm = tm+dt; # updating time for Magnus integrator

# accumulate objf = integral w(t) * ( 1 - | Tr( vSol' * vTarget )/N |^2 )
    infidelity = weightf(tm)*(1 - trace_fid_cmplx(uSol, vTarget, lab_frame, tm, omega));
    objf = objf + dt*infidelity;

# Stromer-Verlet
    infidelity_0 = weightf(t)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega));

    for q=1:stages
# the following call updates ( t, ur, vr)
      [ur, vi, t] = stromer_verlet_mat3(ur, vi, rfunc, ifunc, t, gamma(q)*dt, pcof, H0, amat, Ident, d_omega, zeroMat, zeroMat, zeroMat, zeroMat); 
# real arithmetic for Verlet
# accumulate objf = integral w(t) * ( 1 - |Tr( uSol' * vTarget )/N|^2 )
      infidelity = weightf(t)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega));
      objf_v = objf_v + gamma(q)*dt* 0.5* (infidelity_0 + infidelity);
      infidelity_0 = infidelity;	# save previous values for next stage
    end

# save solutions from both methods to evaluate differences
    if (verbose)
      usave(:,:,step+1) = uSol;
      usaver(:,:,step+1) = ur;
      usavei(:,:,step+1) = -vi;
      
    end # if verbose
  end # for (time stepping loop)

# correct Magnus integration of objective function from last time-step
  infidelity = weightf(tm)*(1 - trace_fid_cmplx(uSol, vTarget, lab_frame, tm, omega));
  objf = objf - 0.5*dt*infidelity;

# no correction needed for the Verlet scheme (objf_v)
  
  if (lab_frame)
    uFinal = uSol;
    uFinal_r = ur;
    uFinal_i = -vi;
  else
    RotMat = diag([ exp(I*omega*T) ]);
    uFinal = RotMat' * uSol;
# verlet needs real arithmetic
    RotMat_r = diag([ cos(omega*T) ]);
    RotMat_i = diag([ sin(omega*T)  ]);
    uFinal_r = RotMat_r' *ur - RotMat_i' * vi;
    uFinal_i = -RotMat_r' * vi - RotMat_i * ur;
  end

				# plot results
  if (verbose)
				# difference at final time
    Nplot = nsteps + 1;
    printf("Difference Verlet-Magnus:\n");
    printf("Initial-data  Component     abs(real)       abs(imag)\n");
    for c=1:Ntot
      for q=1:N
	printf("  %8d  %8d  %15.8e %15.8e\n", q, c, abs(usaver(c,q,Nplot) - real(usave(c,q,Nplot))),  abs(usavei(c,q,Nplot) - imag(usave(c,q,Nplot))) );
      end
    end
				# unitary?
    printf(" Column  Mnrm   Vnrm\n");
    for q=1:N
      Vnrm = usaver(:,q,Nplot)' * usaver(:,q,Nplot) + usavei(:,q,Nplot)' * usavei(:,q,Nplot);
      Vnrm = sqrt(Vnrm);
      Mnrm = norm(usave(:,q,Nplot));
      printf(" %d  %e  %e\n", q, Mnrm, Vnrm);
    end
			    # tmp: compare solutions from both methods

    tplot = linspace(0,T,Nplot);
    c=3;
    q=3;

#    if (lab_frame)
				# real part
      ## figure(1);
      ## h=plot(tplot, real(usave(c,q,:)), "r", tplot, usaver(c,q,:), "b");
      ## tstr = sprintf("Real part, component %d\n", c);
      ## title(tstr);
      ## legend("Re(Magnus)", "Re(Verlet)", "location", "east");
      ## axis tight

      ## figure(2);
      ## h=plot(tplot, real(usave(c,q,:))- usaver(c,q,:));
      ## tstr = sprintf("Difference, component %d\n", c);
      ## title(tstr);
      ## legend("Re(Magnus - Verlet)", "location", "east");
      ## axis tight

      ## 				# imaginary part
      ## figure(3);
      ## h=plot( tplot, imag(usave(c,q,:))- usavei(c,q,:));
      ## tstr = sprintf("Difference, component %d\n", c);
      ## title(tstr);
      ## legend( "Im(Magnus-Verlet)", "location", "east");
      ## axis tight

      ## figure(4);
      ## h=plot( tplot, imag(usave(c,q,:)), "r", tplot, usavei(c,q,:), "b");
      ## tstr = sprintf("Imaginary part, component %d\n", c);
      ## title(tstr);
      ## legend( "Im(Magnus)", "Im(Verlet)", "location", "east");
      ## axis tight
    
#    plotunitary(usaver, T, abs_or_real);
    plotunitary2(usaver+I*usavei, T, abs_or_real);
    
		# evaluate the polynomials at the discrete time levels
		# evaluate all polynomials on the midpoint grid
    td = linspace(0, T, nsteps+1);
    p_r = rfunc(td,pcof);
    p_i = ifunc(td,pcof);
    figure(5);
    clf;
    subplot(2,1,1);
    h=plot(td, p_r,"b-");
    legend("Real");
#    h=plot(td, p_r,"b", td, p_i, "r");
#    legend("Real",  "Imag");
    axis("tight");
    ## pmin = min(ptot);
    ## pmax = max(ptot);
    ## pdelta=pmax-pmin;
    ## axis([0,T,pmin-0.1*pdelta,pmax+0.1*pdelta]);
    set(h,"linewidth",2);
    title("Control function");

    subplot(2,1,2);
    wghf1 = weightf(td);
    h = plot(td, wghf1, "m");
    axis tight;
    set(h,"linewidth",2);
    title("Weight function");
#    legend("e0 & e1", "e2 & e3","location","north");
    ## h = plot( [1:D/2], pcof(1:2:end), "b*",  [1:D/2], pcof(2:2:end), "r*");
    ## set(h,"markersize",10);
    ## amin = min(pcof);
    ## amax= max(pcof);
    ## adelta=amax-amin;
    ## axis([0.5 D/2+0.5, amin-0.1*adelta, amax+0.1*adelta]);
    ## legend("cos", "sin", "location", "east");
    ## title("Parameters");


				# output final solution and target
    printf("uTarget:  id1          id2           id3           id4\n");
    for k=1:Ntot
      printf("row=%d: ", k);
      for j=1:N
	printf(" %13.6e", uTarget(k,j));
      end
      printf("\n");
    end

    printf("uFinal-Ma:  id1          id2           id3           id4\n");
    for k=1:Ntot
      printf("row=%d: ", k);
      for j=1:N
	printf(" %13.6e", abs(uSol(k,j)));
      end
      printf("\n");
    end
    
    printf("uFinal-Ve:  id1          id2           id3           id4\n");
    for k=1:Ntot
      printf("k=%d: ", k);
      for j=1:N
	printf(" %13.6e", abs(uFinal_r(k,j) + I*uFinal_i(k,j))  );
      end
      printf("\n");
    end

# tmp
    ## printf("size(uFinal_r'):")
    ## size(uFinal_r')
    ## printf("size(uTarget):")
    ## size(uTarget)

    uFinal_uTarget = (uFinal_r + I*uFinal_i)' * uTarget;
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

    printf("Forward calculation: Parameter pcof =[ %e", pcof(1));
    if (D>=2)
      for q=2:D
	printf(", %e", pcof(q));
      end
      printf(" ]\n");
    end
				# check if uFinal is unitary
    utest = ctranspose(uFinal) * uFinal - diag(ones(1,N));
    printf("LabFrame = %d, order = %d, Final unitary infidelity = %e, Final |trace| gate fidelity = %e\n", lab_frame, order, norm(utest), final_fidelity);
    printf("Nsteps=%d, Integrated |trace|^2 infidelity: Magnus = %e,  Verlet = %e\n", nsteps, objf, objf_v);
  end # if verbose
end
