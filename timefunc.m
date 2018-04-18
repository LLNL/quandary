%-*-octave-*--
%
% timefunc: setup a linear combination of polynomial functions of time
%
% USAGE:
% 
% [td, pd] = timefunc(a1, nsteps)
%
% INPUT:
% a1: amplitude of the polynomial functions (1-D array, required)
% nsteps: number of time steps (0,1,2,...,nsteps) (optional, default nsteps = 100)
%
% OUTPUT:
%
% td: 1-D array of time values
% pd: 1-D array of P(td, a1)
%
function  [pad, td] = timefunc(D, nsteps)
  
  if nargin < 1
    D=1;
  end

  if nargin < 2
    nsteps=100;
  end

  if D>12
    printf("The number of parameters D= %d exceeds 12 (not currently implemented)\n");
    return;
  end
  
# Final time T
  T = 20;
  td = linspace(0, T, nsteps+1)'; # column vector
  dt = td(2)-td(1);
  printf("Final time = %e, number of time steps = %d, time step = %e\n", ...
	 T, nsteps, dt);
  printf("Number of polynomials D = %d \n", D);

# evaluate the polynomials at the discrete time levels
  pad = zeros(nsteps+1, D);
  pad(:, 1) = (10*(td./T).^3 - 15*(td./T).^4 + 6*(td./T).^5); # first polynomial
  for q=2:D
    if (q==2)
      tp = T;
      t0 = 0.5*T;
    elseif (q>=3 & q<=5)
      tp = 0.5*T;
      t0 = 0.25*(q-2)*T;
    elseif (q>=6 & q<=12)
      tp = 0.25*T;
      t0 = 0.125*(q-5)*T;
    end
    tau = (td - t0)/tp;
    mask = (tau >= -0.5 & tau <= 0.5);
    pad(:,q) = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;
  end # for

end
