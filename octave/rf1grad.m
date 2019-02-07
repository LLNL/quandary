%-*-octave-*--
%
% USAGE:  f = rf1grad(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of the gradient of the time function wrt the pcof parameter vector
%
function  [f] = rf1grad(t, pcof)
  D = size(pcof,1); # parameter dimension
  f =zeros(D, 1);
  
# Final time T
  global T;
# Frequency
  d1 = 24.64579437;
  
  tp = T;
  t0 = 0.5*T;
  
  tau = (t - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);

  f(1) = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3 .*cos(d1*t);

end
