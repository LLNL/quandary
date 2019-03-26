%-*-octave-*--
%
% USAGE:  f = ef16(t, param)
%
% INPUT:
% t: time (real scalar)
% params: struct containing (pcof, T, d_omega) 
%
% OUTPUT:
%
% f: 4x1 vector of the envelope function for each carrier frequency at time t
%
function  [f] = ef16(t, param)
  D = size(param.pcof,1);
  if (D < 16)
    printf("ERROR: ef16 only works when pcof has at least 16 elements\n");
    f=-999;
    return;
  end
  f = zeros(4,length(t));

  # base wavelet
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

# period T/2 wavelets, centered at (0.25, 0.5, 0.75)*T
  tp = 0.5*param.T;

  tc = 0.25*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1,:) = f(1,:) + param.pcof(5) * envelope;
# state 2 to 3
  f(2,:) = f(2,:) + param.pcof(6) * envelope;
# state 3 to 4
  f(3,:) = f(3,:) + param.pcof(7) * envelope;
# state 4 to 5
  f(4,:) = f(4,:) + param.pcof(8) * envelope;


  tc = 0.5*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1,:) = f(1,:) + param.pcof(9) * envelope;
# state 2 to 3
  f(2,:) = f(2,:) + param.pcof(10) * envelope;
# state 3 to 4
  f(3,:) = f(3,:) + param.pcof(11) * envelope;
# state 4 to 5
  f(4,:) = f(4,:) + param.pcof(12) * envelope;

  tc = 0.75*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1,:) = f(1,:) + param.pcof(13) * envelope;
# state 2 to 3
  f(2,:) = f(2,:) + param.pcof(14) * envelope;
# state 3 to 4
  f(3,:) = f(3,:) + param.pcof(15) * envelope;
  # state 4 to 5
  f(4,:) = f(4,:) + param.pcof(16) * envelope;
  
end
