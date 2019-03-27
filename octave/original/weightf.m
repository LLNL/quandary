%-*-octave-*--
%
% USAGE:  w = weightf(t)
%
% INPUT:
% t: time (real scalar)
% T: duration of the simulation
%
% OUTPUT:
%
% w: weight function used by objective function
%
function  [w] = weightf(t, T)

# period
#  tp = T/50;
  tp = T/10;
  xi = 4/tp; # scale factor
  
# center time
  tc = T;
  tau = (t - tc)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);
  w = xi*64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;
end
