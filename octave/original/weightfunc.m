%-*-octave-*--
%
% objective: solve a model problem from quantum control theory
%
% USAGE:
% 
% [cost, ufinal] = objective(a1, verbose, cfl)
%
% INPUT:
% a1: amplitude of control function #1 (default = 1.0)
%
% OUTPUT:
% cost: sum(cfunc): cost functional of the response
% ufinal: state vector at t=T
%
function weightfunc(a1, verbose)

  abs_or_real=0; # plot the abs of the solution (1 for real)

  wconst = 0.01; # for response to e2 and e3
  
  if nargin < 1
    a1 = 1.0;
  end

  if nargin < 2
    verbose=0;
  end

  cfl = 0.25;

  N = 4; # vector dimension

  D = length(a1); # parameter dimension
  
# coefficients in H0
  d0 = 0;
  d1 = 24.64579437;
  d2 = 47.88054868;
  d3 = 69.70426293;

  H0 = diag([d0, d1, d2, d3]);

  ## H1 = I*[0, 1, 0, 0;
  ## 	  -1, 0, sqrt(2), 0;
  ## 	  0, -sqrt(2), 0, sqrt(3);
  ## 	  0, 0, -sqrt(3), 0];

  H1 = [0, 1, 0, 0;
  	1, 0, sqrt(2), 0;
  	0, sqrt(2), 0, sqrt(3);
  	0, 0, sqrt(3), 0];

# final time
  T = 15;
# first evaluate the polynomials on a coarse grid
  pad0 = timefunc(D, 100);
  ptot0 = pad0*a1;
  [pmax imax] = max(ptot0); # assumes ptot is real-valued
  [pmin imin] = min(ptot0);

  nsteps=200;
		# evaluate the polynomials at the discrete time levels
  td = linspace(0,T,nsteps+1)'; # column vector
# evaluate all polynomials on the grid
  pad = timefunc(D, nsteps);
#  pad(:, 1) = (10*(td./T).^3 - 15*(td./T).^4 + 6*(td./T).^5); # first polynomial
# sum up all polynomial components
  ptot = pad*a1;

# form the weight function
  tp = 0.125*T;
  t0 = T;
  tau = (td - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);
  wghf1 = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;

# different weight functions for different components
  wghf = zeros(nsteps+1,N);
  wghf(:,1) = wghf1;
  wghf(:,2) = wghf1;
  wghf(:,3) = (1-wconst)*wghf1+wconst;
  wghf(:,4) = (1-wconst)*wghf1+wconst;

  figure(1)
  h=plot(td, wghf(:,1), td, wghf(:,3)); axis tight
  set(h,"linewidth",2)
  set(gca,"fontsize",16)
  legend("w_0 and w_1", "w_2 and w_3","location","north")
  

end

