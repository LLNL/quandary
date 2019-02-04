%-*-octave-*--
%
% USAGE:  w = weightf(t)
%
% INPUT:
% t: time (real scalar)
%
% OUTPUT:
%
% w: weight function used by objective function
%
function  [w] = weightf(t, pcof)

# Final time T
  T = 15;

# period
  tp = 0.125*T;
# center time
  t0 = T;
  tau = (t - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);
  w = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;
end
