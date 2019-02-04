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
function [objf uFinal] = traceobjfunc(pcof, verbose, order)

  abs_or_real=1; # plot the magnitude (abs) of real part of the solution (1 for real)

  if nargin < 1
    pcof(1) = 1.0;
    pcof(2) = 0.0;
  end

  if nargin < 2
    verbose=0;
  end

  if nargin < 3
    order = 4;
  end

  if (order == 6)
    stages = 9;
  end
  
  cfl = 0.1;

  N = 4; # vector dimension

  D = size(pcof,1); # parameter dimension

  ## if (mod(D,2) == 1)
  ##   printf("ERROR: D=%d, is ODD\n", D);
  ##   return;
  ## end
  
# coefficients in H0
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

  lab_frame = 0;
  if (lab_frame)
# lab frame
    H0 = diag([d0, d1, d2, d3]);
    d_omega = [0, 0, 0, 0];
  else
# rotating frame
    H0 = diag([0, 0, 0, 0]);
    d_omega = [d1-d0, d2-d1, d3-d2, 0];
  end

				# lowering op
  amat = [0, 1, 0, 0;
  	0, 0, sqrt(2), 0;
  	0, 0, 0, sqrt(3);
  	0, 0, 0, 0];
# raising op
  adag = amat';
  
  K1 = [0, 1, 0, 0;
  	1, 0, sqrt(2), 0;
  	0, sqrt(2), 0, sqrt(3);
  	0, 0, sqrt(3), 0];

  S1 = [0, 1, 0, 0;
  	-1, 0, sqrt(2), 0;
  	0, -sqrt(2), 0, sqrt(3);
  	0, 0, -sqrt(3), 0];

#  S1 = zeros(N,N);
# final time
  T = 15;

  if (verbose)
    printf("Vector dim (N) = %d, Param dim (D) = %d, pcof(1) = %e, Final time = %e, CFL = %e\n", N, D, pcof(1), T, cfl);
  end

  if (lab_frame) # max eigenvalue determined by H0 + max(control terms)
    H=H0+pcof(1)*(amat+adag)+I*pcof(2)*(amat-adag);
				# estimate largest eigenvalue
    lambda = eig(H);
    maxeig1 = norm(lambda,"inf");
    if (verbose)
      printf("maxeig1 = %e\n", maxeig1);
    end

    H=H0-pcof(1)*(amat+adag)-I*pcof(2)*(amat-adag);
    lambda = eig(H);
    maxeig2 = norm(lambda,"inf");
    if (verbose)
      printf("maxeig2 = %e\n", maxeig2);
    end
    maxeig = max(maxeig1, maxeig2);
  else
# rotating frame: time step essentially determined by time scale of forcing
    maxeig = max(abs(d_omega))/(2*pi);
  end

		   # Final time T
  dt = cfl/maxeig; # largest eigenvalue of H0 = d3, H0+poly*K1 estimated by maxeig
  nsteps = ceil(T/dt);
  dt = T/nsteps;
  if (verbose)
    printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   T, nsteps, maxeig, cfl, dt);
  end

# different weight functions for different components
  wconst = 0.01; # for response to e2 and e3
  
# the basis for the initial data as a matrix
  Ident=diag([1, 1, 1, 1]);
  U0 = Ident;

# Target state at t=T (always real)
  uTarget = [0, 1, 0, 0;
	     1, 0, 0, 0;
	     0, 0, 1, 0;
	     0, 0, 0, 1];
	
  if (lab_frame)
    vTarget = uTarget;
# uTarget is real
    vTarget_r = uTarget;
    vTarget_i = zeros(4,4);
  else  # target in rotating frame
    RotMat = diag([ exp(I*d0*T), exp(I*d1*T), exp(I*d2*T), exp(I*d3*T) ]);
    vTarget = RotMat*uTarget;
				# real arithmetic for Verlet
    RotMat_r = diag([ cos(d0*T), cos(d1*T), cos(d2*T), cos(d3*T) ]);
    RotMat_i = diag([ sin(d0*T), sin(d1*T), sin(d2*T), sin(d3*T) ]);
# uTarget is real
    vTarget_r = RotMat_r*uTarget;
    vTarget_i = RotMat_i*uTarget;
  end

# initial data and allocation of solution vectors
  uSol = U0;
# real and negative imaginary part of the solution
  ur = U0;
  vi = zeros(N, N);

# RK stage variables
  kay1 = zeros(N,N);
  kay2 = zeros(N,N);
  ell1 = zeros(N,N);
  ell2 = zeros(N,N);

# zero forcing
  fv_0 = zeros(N,N);
  fv_1 = zeros(N,N);
  fu_1o2 = zeros(N,N);

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
  
# testing octave syntax
  ## step=1;
  ## S = qtot(step).*S1; # skew-symmetric part
  ## rhs = H*ur + S*vi;
  ## l1 = linsolve( Ident-0.5*dt*S, rhs ); # - 0.5*dt*S);
  ## printf("Size(Ident)= (%d, %d), size(rhs)=(%d,%d), size(l1)=(%d,%d)\n", size(Ident), size(rhs), size(l1));#
# end test

  v1 = zeros(N,1);
  v2 = zeros(N,1);

			     # time stepping loop, harmonic oscillator
  if (verbose)
    usave = zeros(N,N,nsteps+1);
    usaver = zeros(N,N,nsteps+1);
    usavei = zeros(N,N,nsteps+1);
    usave(:,:,1) = uSol;
    usaver(:,:,1) = ur;
    usavei(:,:,1) = -vi;
  end

# handles to time and forcing functions
  rfunc = @rf1;
  ifunc = @if1;
  uforce = @uzero;
  vforce = @vzero;  
  
  separable = (norm(S1) < 1e-15);
  printf("Separable = %d, order of accuracy = %d\n", separable, order);

# for computing the objf function
  objf = 0;
  objf_v = 0;

  t=0;
  tm=0;
  step=0;
  for step=1:nsteps
# 2nd order Magnus integrator
#    dmat = expm(-I*diag(d_omega.*(tm+0.5*dt)));
    dmat_r_1o2 = diag([ cos(d_omega(1)*(tm+0.5*dt)), cos(d_omega(2)*(tm+0.5*dt)), cos(d_omega(3)*(tm+0.5*dt)), cos(d_omega(4)*(tm+0.5*dt)) ]);
    dmat_i_1o2 = diag([ -sin(d_omega(1)*(tm+0.5*dt)), -sin(d_omega(2)*(tm+0.5*dt)), -sin(d_omega(3)*(tm+0.5*dt)), -sin(d_omega(4)*(tm+0.5*dt)) ]);

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

# accumulate objf = integral w(t) * ( 1 - |Tr( uSol' * vTarget )/N|^2 )
    fidelity = trace(ctranspose(uSol) * vTarget)/N;
    objf = objf + dt*weightf(tm)*( 1 - abs(fidelity)^2 );

# Stromer-Verlet
    for q=1:stages
      ur0=ur;
      vi0=vi;
      t0=t;
      [ur, vi, t] = stromer_verlet_mat(ur, vi, rfunc, ifunc, t, gamma(q)*dt, pcof, H0, amat, adag, Ident, d_omega, uforce, vforce); # t, ur, vr are updated
# real arithmetic for Verlet
# accumulate objf = integral w(t) * ( 1 - |Tr( uSol' * vTarget )/N|^2 )
      ua = 0.5*(ur+ur0);
      va = 0.5*(vi+vi0);
      fidelity2 = (trace(ua' * vTarget_r - va' * vTarget_i)/N)^2 + (trace(ua' * vTarget_i + va' * vTarget_r)/N)^2;
      ta = 0.5*(t0+t);
      objf_v = objf_v + gamma(q)*dt*weightf(ta)*( 1 - fidelity2 );
    end

# save solutions from both methods to evaluate differences
    if (verbose)
      usave(:,:,step+1) = uSol;
      usaver(:,:,step+1) = ur;
      usavei(:,:,step+1) = -vi;
      
    end # if verbose
  end # for (time stepping loop)

# correct integration of objective function from last time-step
  fidelity = trace(ctranspose(uSol) * vTarget)/N;
  objf = objf - 0.5*dt*weightf(T)*( 1 - abs(fidelity)^2 );

# no correction needed for the Verlet scheme (objf_v)
  
  if (lab_frame)
    uFinal = uSol;
    uFinal_r = ur;
    uFinal_i = -vi;
  else
    RotMat = diag([ exp(I*d0*T), exp(I*d1*T), exp(I*d2*T), exp(I*d3*T) ]);
    uFinal = RotMat' * uSol;
# verlet needs real arithmetic
    RotMat_r = diag([ cos(d0*T), cos(d1*T), cos(d2*T), cos(d3*T) ]);
    RotMat_i = diag([ sin(d0*T), sin(d1*T), sin(d2*T), sin(d3*T) ]);
    uFinal_r = RotMat_r' *ur - RotMat_i' * vi;
    uFinal_i = -RotMat_r' * vi - RotMat_i * ur;
  end

				# plot results
  if (verbose)
				# difference at final time
    Nplot = length(usave(1,1,:));
    printf("Difference Verlet-Magnus:\n");
    printf("Initial-data  Component     abs(real)       abs(imag)\n");
    for q=1:N
      for c=1:N
	printf("  %8d  %8d  %15.8e %15.8e\n", q, c, abs(usaver(c,q,Nplot) - real(usave(c,q,Nplot))),  abs(usavei(c,q,Nplot) - imag(usave(c,q,Nplot))) );
      end
    end
				# unitary?
    printf(" Initial data  Mnrm   Vnrm\n");
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
    plotunitary(usave, T, abs_or_real);
    
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
    ## pmin = min(ptot);
    ## pmax = max(ptot);
    ## pdelta=pmax-pmin;
    ## axis([0,T,pmin-0.1*pdelta,pmax+0.1*pdelta]);
    set(h,"linewidth",2);
    title("Forcing function");

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
  end

				# output final solution and target
  printf("uTarget:  id1          id2           id3           id4\n");
  for k=1:N
    printf("k=%d: ", k);
    for j=1:N
      printf(" %13.6e", uTarget(j,k));
    end
    printf("\n");
  end

  printf("uSol-Ma:  id1          id2           id3           id4\n");
  for k=1:N
    printf("k=%d: ", k);
    for j=1:N
      printf(" %13.6e", abs(uSol(j,k)));
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
  final_Infidelity = 1 - abs(trace(ctranspose(uFinal) * uTarget)/N);

  if (verbose)
    printf("Forward calculation: Parameter pcof =[ %e", pcof(1));
    for q=2:D
      printf(", %e", pcof(q));
    end
    printf(" ]\n");
				# check if uFinal is unitary
    utest = ctranspose(uFinal) * uFinal - U0;
    printf("Final unitary infidelity = %e, Final |trace| gate infidelity = %e\n", norm(utest), final_Infidelity);
    printf("Nsteps=%d, Integrated |trace|^2 infidelity: Magnus = %e,  Verlet = %e\n", nsteps, objf, objf_v);
  end # if verbose
end
