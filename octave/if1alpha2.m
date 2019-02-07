%-*-octave-*--
%
% USAGE:  f = if1alpha2(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: imaginary part of the derivative of the time function wrt pcof(2)
%
function  [f] = if1alpha2(t, pcof)

# Final time T
  global T;
# Frequency
  d1 = 24.64579437;
  
  tp = T;
  t0 = 0.5*T;
  
  tau = (t - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);

  f = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3 .*sin(d1*t);

end
