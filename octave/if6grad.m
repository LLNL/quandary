%-*-octave-*--
%
% USAGE:  f = if6grad(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of the gradient of the time function wrt the pcof parameter vector
%
function  [f] = if6grad(t, pcof)
  D = size(pcof,1);
  if (D != 6)
    printf("ERROR: rf6 only works when D=6!\n");
    f=-999;
    return;
  end
  f =zeros(D, 1);

# Final time T
  global T;
# Frequency
# coefficients in H0
  d_omega = zeros(1,3);
  d_omega(1) = 24.64579437;
  d_omega(2) = 47.88054868 - 24.64579437;
  d_omega(3) = 69.70426293 - 47.88054868;

  # base wavelet
  tp = T;
  tc = 0.5*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(2) = envelope .*sin(d_omega(1)*t);
# state 2 to 3
  f(4) = envelope .*sin(d_omega(2)*t);
# state 3 to 4
  f(6) = envelope .*sin(d_omega(3)*t);
  
end
