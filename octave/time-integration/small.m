%-*-octave-*--
%
% small: integrate  Schroedinger's equation for a small model with exact solution
%
% USAGE:
% 
% [] = small(cfl, testcase, order, stages)
%
% INPUT:
% cfl: dt = maxeig/cfl
% testcase:  (default is 1) (all cases have an exact solution): 
%               0: Non-separable Hamiltonian
%               1: Separable Hamiltonian, 
%               2: Polynomial time function, separable Hamitonian;
%               3: Polynomial time function, non-separable Hamitonian;
% order: order of accuracy, may be 2, 4, or 6.
% stages: only used for order = 6, in which case it is 7 or 9 (default)
%
% OUTPUT:
% err: error in solution
%
function [err] = small(cfl, testcase, order, stages)

  ploterr=1; # Plot the error (1) or the solution (0)?

  if nargin < 1
    cfl=0.1;
  end

  if nargin < 2
    testcase=1;
  end

  if nargin < 3
    order=4;
  end

  if nargin < 4
    stages=-1;
  end
  

  verbose = 1;

  N = 2; # vector dimension

  ## if (mod(D,2) == 1)
  ##   printf("ERROR: D=%d, is ODD\n", D);
  ##   return;
  ## end
  
  if (testcase==1 || testcase==2)
    K1 = [0, 1; 1, 0];
    S1 = [0, 0; 0, 0];
  elseif (testcase==0 || testcase==3)
    K1 = [0, 0; 0, 0];
    S1 = [0, 1; -1, 0];
  end

				# final time
  period = 1;
  omega = 2*pi/period;

  T = 5*pi;


  if (verbose)
    printf("Testcase = %d, Final time = %e, CFL = %e\n", testcase, T, cfl);
  end

  lambda = eig(K1+I*S1);
  maxeig = norm(lambda,"inf");
  if (verbose)
    printf("maxeig = %e\n", maxeig);
  end

  cfunc = zeros(N,1);
  beta = zeros(N,1);
  cu_sp = zeros(N,1);
  ca1 = zeros(N,1);
  vect = zeros(N,1);

				# Final time T
  dt = cfl/maxeig; 
  nsteps = ceil(T/dt);
  dt = T/nsteps;
  if (verbose)
    printf("Final time = %e, number of time steps = %d, max eigenvalue = %e, cfl = %e, time step = %e\n", ...
	   T, nsteps, maxeig, cfl, dt);
  end

			  # the basis for the initial data as a matrix
  Ident=diag([1, 1]);

  U0 = [1;0]; # works for both separable and non-separable Hamiltonians

		     # initial data and allocation of solution vectors
  uSol = U0;
		    # real and negative imaginary part of the solution
  ur = U0;
  vi = zeros(N, 1);


				# testing octave syntax
  ## t=0.5
  ## H = K1.*0.5*sin(0.5*omega*t)^2; # symmetric part
  ## rhs = H*ur + S1*vi;
  ## ell1 = linsolve( Ident-0.5*dt*S1, rhs ); # - 0.5*dt*S1);
  ## printf("Size(Ident)= (%d, %d), size(rhs)=(%d,%d), size(ell1)=(%d,%d)\n", size(Ident), size(rhs), size(ell1));
  ## return
				# end test

  v1 = zeros(N,1);
  v2 = zeros(N,1);
				# for computing the cost function
  cost = 0;

			     # time stepping loop, harmonic oscillator
  if (verbose)
    usave = zeros(N,nsteps+1);
    usaver = zeros(N,nsteps+1);
    usavei = zeros(N,nsteps+1);
    usave(:,1) = uSol;
    usaver(:,1) = ur;
    usavei(:,1) = -vi;
  end

			
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
  t=0;
  step=0;

  if (testcase==1)
    tfunc   = @tfunc1;
    uforce = @uforce1;
    vforce = @vforce1;
  elseif (testcase==0)
    tfunc   = @tfunc0;
    uforce = @uforce0;
    vforce = @vforce0;
  elseif (testcase==2)
    tfunc   = @tfunc2;
    uforce = @uforce2;
    vforce = @vforce2;
  elseif (testcase==3)
    tfunc   = @tfunc3;
    uforce = @uforce3;
    vforce = @vforce3;
  else
    printf("ERROR: unknown testcase = %d\n", testcase);
    return;
  end

  separable = (norm(S1) < 1e-15);
  printf("Separable = %d\n", separable);
  
  for step=1:nsteps
    tf_1o2 = tfunc(t+0.5*dt,omega);

				# 2nd order Magnus integrator
    uSol = expm(-I*dt*tf_1o2*(K1+I*S1)) * uSol;

    for q=1:stages
      [ur, vi, t] = stromer_verlet(ur, vi, tfunc, t, gamma(q)*dt, omega, K1, S1, Ident, separable, uforce, vforce); # t, ur, vr are updated
    end

				# evaluate energy
    if (verbose)
      usave(:,step+1) = uSol;
      usaver(:,step+1) = ur;
      usavei(:,step+1) = -vi;
    end # if verbose
  end # for (time stepping loop)

  uFinal = uSol;
  uFinalr = ur;
  uFinali = -vi;
				# plot results
  if (verbose)
				# difference at final time
    Nplot = length(usave(1,:));
    tplot = linspace(0,T,Nplot);
    if (testcase==1 || testcase==2 || testcase==3)
      phi = 0.25*(tplot - 1/omega*sin(omega*tplot));
      cg = cos(phi);
      ce = -I*sin(phi);
    elseif (testcase==0)
      phi = 0.25*( tplot + 1/omega*(cos(omega*tplot) - 1) );
      cg = cos(phi);
      ce = -sin(phi);
    end

    ## printf("Final solution:\n");
    ## printf("Exact : Re(cg)=%e  Im(cg)=%e\n", real(usave(1,Nplot)), imag(usave(1,Nplot)));
    ## printf("Magnus: Re(cg)=%e  Im(cg)=%e\n", real(cg(Nplot)), imag(cg(Nplot)));
    ## printf("Exact : Re(ce)=%e  Im(ce)=%e\n", real(usave(2,Nplot)), imag(usave(2,Nplot)));
    ## printf("Magnus: Re(ce)=%e  Im(ce)=%e\n", real(ce(Nplot)), imag(ce(Nplot)));
    m_cg_err = abs(usave(1,Nplot) - cg(Nplot));
    m_ce_err = abs(usave(2,Nplot) - ce(Nplot));
    cg_err = sqrt( (usaver(1,Nplot)-real(cg(Nplot)))^2 + (usavei(1,Nplot)-imag(cg(Nplot)))^2 );
    ce_err = sqrt( (usaver(2,Nplot)-real(ce(Nplot)))^2 + (usavei(2,Nplot)-imag(ce(Nplot)))^2 );
    printf("CFL=%g Solution errors:\n  Verlet: cg-err=%e  ce-err=%e\n", cfl, cg_err, ce_err );
    if (testcase==0 || testcase==1)
      printf("  Magnus: cg-err=%e  ce-err=%e\n", m_cg_err, m_ce_err );
    end
    
				# plot solution or error?
    if (ploterr)
				# component 1&2
      figure(1);
      h=plot(tplot,  usaver(1,:)-real(cg), "r", tplot, usavei(1,:)-imag(cg), "r--", tplot, usaver(2,:)-real(ce), "m", tplot, usavei(2,:)-imag(ce), "m--");
      set(gca,"fontsize",16);
      set(h,"linewidth",2);
      tstr = sprintf("Error, test %d, order = %d, stages = %d\n", testcase, order, stages );
      title(tstr);
      legend( "Re(err-1)", "Im(err-1)", "Re(err-2)", "Im(err-2)", "location", "south");
      axis tight

    else
				# plot analytical solution
				# component 1&2, real & imag parts
      figure(1);
      c=1;
      h=plot(tplot, real(cg), "k", tplot, imag(cg), "k--", tplot, real(ce), "r", tplot, imag(ce), "r--");
      set(gca,"fontsize",16);
      set(h,"linewidth",2);
      if (testcase==1)
	tstr = sprintf("Separable Hamiltonian (testcase=1)");
      elseif (testcase==0)
	tstr = sprintf("Non-separable Hamiltonian (testcase=0)");
      elseif (testcase==2)
	tstr = sprintf("Polynomial time function, separable (testcase=2)");
      elseif (testcase==3)
	tstr = sprintf("Polynomial time function, non-separable (testcase=3)");
      end
      title(tstr);
      legend("Re(Comp 1)", "Im(Comp 1)", "Re(Comp 2)", "Im(Comp 2)", "location", "north");
      axis tight
      xlabel("Time")
      ylabel("a.u.");

    end      
  end

  if (verbose)
				# check if uFinal is unitary
				# unitary?
    Vnrm = uFinalr' * uFinalr + uFinali' * uFinali;
    Vnrm = sqrt(Vnrm);
    Mnrm = norm(uFinal);
    printf("Final solution norm: Magnus = %e, Verlet = %e\n", Mnrm, Vnrm);

  end # if verbose
end
