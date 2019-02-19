%-*-octave-*--
%
% USAGE:  w = penalf(t)
%
% INPUT:
% t: time (real scalar)
% T: duration of the simulation
%
% OUTPUT:
%
% w: penalty function used to discourage forbidden states in the objective function
%
function  [w] = penalf(t, T)

				# overall amplitude
# constant part
  const = 1.0/T;
  alpha = 1;
# period
#  tp = T/50;
  tp = T/10;
  xi = 4/tp; # scale factor for wavelet (integral over half is tp/4)

# center time
  tc = T;
  tau = (t - tc)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);

# weigh the constant and wavelet parts such that max w = xi
  w = alpha * const + (1-alpha)*xi* 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3;
end
