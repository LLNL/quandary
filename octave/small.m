%-*-octave-*--
%
% traceobjfunc: solve a model problem from quantum control theory
%
% USAGE:
% 
% [] = small(cfl, separable)
%
% INPUT:
% cfl: dt = maxeig/cfl
% separable: 1: Separable Hamiltonian, 0: Non-separable Hamiltonian (default is 1)
%
% OUTPUT:
% err: error in solution
%
function [err] = small(cfl, separable)

  ploterr=1; # Plot the error (1) or the solution (0)?

  if nargin < 1
    cfl=0.1;
  end

  if nargin < 2
    separable=1;
  end

  verbose = 1;

  N = 2; # vector dimension

  ## if (mod(D,2) == 1)
  ##   printf("ERROR: D=%d, is ODD\n", D);
  ##   return;
  ## end
  
  if (separable)
    K1 = [0, 1; 1, 0];
    S1 = [0, 0; 0, 0];
  else
    K1 = [0, 0; 0, 0];
    S1 = [0, 1; -1, 0];
  end

				# final time
  period = 1;
  omega = 2*pi/period;

  T = 5*pi;


  if (verbose)
    printf("Final time = %e, CFL = %e\n", T, cfl);
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

				# RK stage variables
  k1 = zeros(N,1);
  k2 = zeros(N,1);
  ell1 = zeros(N,1);
  ell2 = zeros(N,1);

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

  t=0;
  step=0;
  for step=1:nsteps
    # evaluate time functions
    if (separable)
      tf_0 = 0.5*(sin(0.5*omega*(t)))^2;
      tf_1o2=0.5*(sin(0.5*omega*(t+0.5*dt)))^2;
      tf_1 =0.5*(sin(0.5*omega*(t+dt)))^2;
    else
      tf_0 = 0.25*(1-sin(omega*(t)));
      tf_1o2 = 0.25*(1-sin(omega*(t+0.5*dt)));
      tf_1 = 0.25*(1-sin(omega*(t+dt)));
    end      

				# 2nd order Magnus integrator
    uSol = expm(-I*dt*tf_1o2*(K1+I*S1)) * uSol;

		    # Partitioned 2nd order RK method (Stromer-Verlet)
# solving for ell1 
    rhs = tf_0*(K1*ur + S1*vi);
    ell1 = linsolve( Ident-0.5*dt*tf_0*S1, rhs );
    k1 = tf_1o2 * (S1*ur - K1*(vi+0.5*dt*ell1) );
    rhs = tf_1o2* (S1*(ur+0.5*dt*k1) - K1*(vi+0.5*dt*ell1) );
    k2 = linsolve( Ident-0.5*dt*tf_1o2*S1, rhs );
    ell2 = tf_1* ( K1*(ur+0.5*dt*(k1+k2)) + S1*(vi+0.5*dt*ell1) );

#S1=0 -> fully explicit
    ## ell1 = tf_0*K1*ur;
    ## k1 = - tf_1o2*K1*(vi+0.5*dt*ell1);
    ## k2 = - tf_1o2*K1*(vi+0.5*dt*ell1);
    ## ell2 = tf_1*K1*(ur+0.5*dt*(k1+k2));
    
    ur = ur + 0.5*dt*(k1 + k2);
    vi = vi + 0.5*dt*(ell1 + ell2);
    t = t+dt;

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
    if (separable)
      phi = 0.25*(tplot - 1/omega*sin(omega*tplot));
      cg = cos(phi);
      ce = -I*sin(phi);
    else
      phi = 0.25*( tplot + 1/omega*(cos(omega*tplot) - 1) );
      cg = cos(phi);
      ce = -sin(phi);
    end

    cg_err = sqrt( (usaver(1,Nplot)-real(cg(Nplot)))^2 + (usavei(1,Nplot)-imag(cg(Nplot)))^2 );
    ce_err = sqrt( (usaver(2,Nplot)-real(ce(Nplot)))^2 + (usavei(2,Nplot)-imag(ce(Nplot)))^2 );
    printf("CFL=%g,  cg-err=%e  ce-err=%e\n", cfl, cg_err, ce_err );
    
				# plot solution or error?
    if (ploterr)
				# component 1
      figure(1);
      c=1;
      h=plot(tplot, real(usave(c,:)-cg), "b", tplot, usaver(c,:)-real(cg), "r",...
	    tplot, imag(usave(c,:)-cg), "b--", tplot, usavei(c,:)-imag(cg), "r--");
      tstr = sprintf("Error, component %d\n", c);
      title(tstr);
      legend("Re(Magnus)-err",  "Re(Verlet)-err", "Im(Magnus)-err",  "Im(Verlet)-err", "location", "north");
      axis tight

				# component 2
      c=2;
      figure(2);
      h=plot(tplot, real(usave(c,:)-ce), "m", tplot, usaver(c,:)-real(ce), "c",...
	    tplot, imag(usave(c,:)-ce), "m--", tplot, usavei(c,:)-imag(ce), "c--");
      tstr = sprintf("Error component %d\n", c);
      title(tstr);
#    legend( "Analytical", "Im(Magnus)", "Verlet", "location", "north");
      legend( "Re(Magnus)-err", "Re(Verlet)-err", "Im(Magnus)-err", "Im(Verlet)-err", "location", "north");
      axis tight

    else
				# plot solution
				# component 1, real & imag parts
      figure(1);
      c=1;
      h=plot(tplot, real(cg), "k", tplot, real(usave(c,:)), "b", tplot, usaver(c,:), "r",...
	     tplot, imag(cg), "k--", tplot, imag(usave(c,:)), "b--", tplot, usavei(c,:), "r--");
#      h=plot(tplot, cg, "k");
      tstr = sprintf("Component %d\n", c);
      title(tstr);
      legend("Re(Analytical)", "Re(Magnus)",  "Re(Verlet)", ...
	     "Im(Analytical)", "Im(Magnus)",  "Im(Verlet)", "location", "north");
      axis tight

				# component 2, real & imag parts
      c=2;
      figure(2);
      h=plot(tplot, real(ce), "k", tplot, real(usave(c,:)), "m", tplot, usaver(c,:), "c",...
	     tplot, imag(ce), "k--", tplot, imag(usave(c,:)), "m--", tplot, usavei(c,:), "c--");
#      h=plot(tplot, ce, "k");
      tstr = sprintf("Component %d\n", c);
      title(tstr);
      legend( "Re(Analytical)", "Re(Magnus)", "Re(Verlet)", ...
	      "Im(Analytical)", "Im(Magnus)", "Im(Verlet)", "location", "north");
      axis tight
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
