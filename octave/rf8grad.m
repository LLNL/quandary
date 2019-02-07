%-*-octave-*--
%
% USAGE:  f = rf8grad(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of the gradient of the time function wrt the pcof parameter vector
%
function  [f] = rf8grad(t, pcof)
  D = size(pcof,1); # parameter dimension
  if (D != 8)
    printf("ERROR: rf8 only works when D=8!\n");
    f=-999;
    return;
  end
  f =zeros(D, 1);
  
# Final time T
  global T;
# Frequency
  d1 = 24.64579437;
  
  tp = T;
  tc = 0.5*T;
  
  tau = (t - tc)/tp;
  f(1) = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

# period T/2 wavelets, centered at (0.25, 0.5, 0.75)*T
  tp = 0.5*T;

  tc = 0.25*T;
  tau = (t - tc)/tp;
  f(3) = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

  tc = 0.5*T;
  tau = (t - tc)/tp;
  f(5) = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

  tc = 0.75*T;
  tau = (t - tc)/tp;
  f(7) = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

# from state 1 (ground) to state 2
  f = f.*cos(d1*t);
  
end
