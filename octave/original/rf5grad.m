%-*-octave-*--
%
% USAGE:  f = rf5grad(t, param)
%
% INPUT:
% t: time (real scalar)
% params: struct containing (pcof, T, d_omega) 
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = rf5grad(t, param)
  D = size(param.pcof,1);
  if (D != 5)
    printf("ERROR: rf5grad only works when pcof has 5 elements\n");
    f=-999;
    return;
  end
  f = zeros(D,1);

  # base wavelet
  tp = param.T;
  tc = 0.5*param.T;
  ## tau = (t - tc)/tp;
  ## envelope = 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;
  xi = (t - tc)/tp;
  envelope = (xi >= -0.5 & xi <= -1/6) .* (9/8 + 4.5*xi + 4.5*xi.^2);
  envelope = envelope + (xi > -1/6 & xi <= 1/6) .* (0.75 - 9*xi.^2);
  envelope = envelope + (xi >  1/6 & xi <= 0.5) .* (9/8 - 4.5*xi + 4.5*xi.^2);
# from state 1 (ground) to state 2
  f(1) = envelope .*cos(param.d_omega(1)*t);
# state 2 to 3
  f(2) = envelope .*cos(param.d_omega(2)*t);
# state 3 to 4
  f(3) = envelope .*cos(param.d_omega(3)*t);
# state 3 to 4
  f(4) = envelope .*cos(param.d_omega(4)*t);
# state 3 to 4
  f(5) = envelope .*cos(param.d_omega(5)*t);

end
