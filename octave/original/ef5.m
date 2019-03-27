%-*-octave-*--
%
% USAGE:  f = ef5(t, param)
%
% INPUT:
% t: time (real scalar)
% params: struct containing (pcof, T, d_omega) 
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = ef5(t, param)
  D = size(param.pcof,1);
  if (D != 5)
    printf("ERROR: ef5 only works when pcof has 5 elements\n");
    f=-999;
    return;
  end
  f = zeros(5,length(t));

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
  f(1,:) = param.pcof(1) * envelope;
# state 2 to 3
  f(2,:) = param.pcof(2) * envelope;
# state 3 to 4
  f(3,:) = param.pcof(3) * envelope;
# state 4 to 5
  f(4,:) = param.pcof(4) * envelope;
# state 5 to 6
  f(5,:) = param.pcof(5) * envelope;
  
end
