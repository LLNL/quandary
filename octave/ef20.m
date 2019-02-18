%-*-octave-*--
%
% USAGE:  f = ef20(t, param)
%
% INPUT:
% t: time (real scalar)
% params: struct containing (pcof, T, d_omega) 
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = ef20(t, param)
  D = size(param.pcof,1);
  if (D != 20)
    printf("ERROR: ef20 only works when pcof has 20 elements\n");
    f=-999;
    return;
  end
  f = zeros(5,length(t));

# period T wavelet,
  tp = param.T;

  tc = 0.5*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;
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

  # period T/2 wavelets, centered at (0.25, 0.5, 0.75)*T
  tp = 0.5*param.T;

  tc = 0.25*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1,:) = f(1,:) + param.pcof(6) * envelope;
# state 2 to 3
  f(2,:) = f(2,:) + param.pcof(7) * envelope;
# state 3 to 4
  f(3,:) = f(3,:) + param.pcof(8) * envelope;
# state 4 to 5
  f(4,:) = f(4,:) + param.pcof(9) * envelope;
# state 5 to 6
  f(5,:) = f(5,:) + param.pcof(10) * envelope;

  tc = 0.5*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1,:) = f(1,:) + param.pcof(11) * envelope;
# state 2 to 3
  f(2,:) = f(2,:) + param.pcof(12) * envelope;
# state 3 to 4
  f(3,:) = f(3,:) + param.pcof(13) * envelope;
  # state 4 to 5
  f(4,:) = f(4,:) + param.pcof(14) * envelope;
# state 5 to 6
  f(5,:) = f(5,:) + param.pcof(15) * envelope;

  tc = 0.75*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1,:) = f(1,:) + param.pcof(16) * envelope;
# state 2 to 3
  f(2,:) = f(2,:) + param.pcof(17) * envelope;
# state 3 to 4
  f(3,:) = f(3,:) + param.pcof(18) * envelope;
  # state 4 to 5
  f(4,:) = f(4,:) + param.pcof(19) * envelope;
# state 5 to 6
  f(5,:) = f(5,:) + param.pcof(20) * envelope;

end
