%-*-octave-*--
%
% USAGE:  f = rf20grad(t, param)
%
% INPUT:
% t: time (real scalar)
% params: struct containing (pcof, T, d_omega) 
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = rf20grad(t, param)
  D = size(param.pcof,1);
  if (D != 20)
    printf("ERROR: rf20grad only works when pcof has 20 elements\n");
    f=-999;
    return;
  end
  f = zeros(D,1);

  # base wavelet
  tp = param.T;
  tc = 0.5*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1) = envelope .*cos(param.d_omega(1)*t);
# state 2 to 3
  f(2) = envelope .*cos(param.d_omega(2)*t);
# state 3 to 4
  f(3) = envelope .*cos(param.d_omega(3)*t);
# state 4 to 5
  f(4) = envelope .*cos(param.d_omega(4)*t);
# state 5 to 6
  f(5) = envelope .*cos(param.d_omega(5)*t);

  # period T/2 wavelets, centered at (0.25, 0.5, 0.75)*T
  tp = 0.5*param.T;

  tc = 0.25*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(6) = envelope .*cos(param.d_omega(1)*t);
# state 2 to 3
  f(7) = envelope .*cos(param.d_omega(2)*t);
# state 3 to 4
  f(8) = envelope .*cos(param.d_omega(3)*t);
# state 4 to 5
  f(9) = envelope .*cos(param.d_omega(4)*t);
# state 5 to 6
  f(10) = envelope .*cos(param.d_omega(5)*t);


  tc = 0.5*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(11) = envelope .*cos(param.d_omega(1)*t);
# state 2 to 3
  f(12) = envelope .*cos(param.d_omega(2)*t);
# state 3 to 4
  f(13) = envelope .*cos(param.d_omega(3)*t);
# state 4 to 5
  f(14) = envelope .*cos(param.d_omega(4)*t);
# state 5 to 6
  f(15) = envelope .*cos(param.d_omega(5)*t);

  tc = 0.75*param.T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(16) = envelope.*cos(param.d_omega(1)*t);
# state 2 to 3
  f(17) = envelope.*cos(param.d_omega(2)*t);
# state 3 to 4
  f(18) = envelope.*cos(param.d_omega(3)*t);
  # state 4 to 5
  f(19) = envelope.*cos(param.d_omega(4)*t);
# state 5 to 6
  f(20) = envelope .*cos(param.d_omega(5)*t);

end
