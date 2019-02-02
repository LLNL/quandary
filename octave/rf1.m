%-*-octave-*--
%
% USAGE:  f = rf1(t, acof)
%
% INPUT:
% t: time (real scalar)
% acof: coefficient
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = rf1(t, acof)

# Final time T
  T = 15;
# Frequency
  d1 = 24.64579437;
  
  tp = T;
  t0 = 0.5*T;
  
  tau = (t - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);

  f = acof(1) * 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3 .*cos(d1*t);

end
