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

  abs_or_real=1; # plot the magnitude (abs) of real part of the solution (1 for real)

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
  
  cfl = 0.1;

  N = 4; # vector dimension

  D = size(pcof,1); # parameter dimension
  
# coefficients in H0
  omega = zeros(1,4);
  omega(1) = 0;
  omega(2) = 24.64579437;
  omega(3) = 47.88054868;
  omega(4) = 69.70426293;

  lab_frame = 0;
##   if (lab_frame)
## # lab frame
##     H0 = diag([omega(1), omega(2), omega(3), omega(4)]);
##     d_omega = [0, 0, 0, 0];
##   else
# rotating frame
  H0 = diag([0, 0, 0, 0]);
  d_omega = [omega(2)-omega(1), omega(3)-omega(2), omega(4)-omega(3), 0];
##  end

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
  
# the basis for the initial data as a matrix
  Ident=diag([1, 1, 1, 1]);
  U0 = Ident;

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

# initial data and allocation of solution vectors
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
  
  v1 = zeros(N,1);
  v2 = zeros(N,1);

			     # time stepping loop, harmonic oscillator
  if (verbose)
    usaver = zeros(N,N,nsteps+1);
    usavei = zeros(N,N,nsteps+1);
    usaver(:,:,1) = ur;
    usavei(:,:,1) = -vi;
  end

# handles to time and forcing functions
  rfunc = @rf1;
  ifunc = @if1;
  uforce = @uzero;
  vforce = @vzero;  
  
  separable = 0;

# for computing the objf function
  objf_v = 0;

  t=0;
  tm=0;
  step=0;
  for step=1:nsteps

# Stromer-Verlet
    infidelity_0 = weightf(t)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega));

    for q=1:stages
      [ur, vi, t] = stromer_verlet_mat(ur, vi, rfunc, ifunc, t, gamma(q)*dt, pcof, H0, amat, adag, Ident, d_omega, uforce, vforce); # t, ur, vr are updated
# real arithmetic for Verlet
# accumulate objf = integral w(t) * ( 1 - |Tr( uSol' * vTarget )/N|^2 )
      infidelity = weightf(t)*(1-trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega));
      objf_v = objf_v + gamma(q)*dt* 0.5* (infidelity_0 + infidelity);
      infidelity_0 = infidelity;	# save previous values for next stage
    end

# save solutions from both methods to evaluate differences
    if (verbose)
      usaver(:,:,step+1) = ur;
      usavei(:,:,step+1) = -vi;
      
    end # if verbose
  end # for (time stepping loop)

# verlet needs real arithmetic
  RotMat_r = diag([ cos(omega(1)*T), cos(omega(2)*T), cos(omega(3)*T), cos(omega(4)*T) ]);
  RotMat_i = diag([ sin(omega(1)*T), sin(omega(2)*T), sin(omega(3)*T), sin(omega(4)*T) ]);
  uFinal_r = RotMat_r' *ur - RotMat_i' * vi;
  uFinal_i = -RotMat_r' * vi - RotMat_i * ur;

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
    
    plotunitary(usaver, T, abs_or_real);
    
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
    printf("Nsteps=%d, Integrated |trace|^2 infidelity:  Verlet = %e\n", nsteps, objf_v);
  end # if verbose
end
