%-*-octave-*--
%
% USAGE:  f = rf18grad(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = rf18grad(t, pcof)
  D = size(pcof,1);
  if (D != 18)
    printf("ERROR: rf18 only works when D=18!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);

# Final time T
  global T;
# Frequencies
  omega = [ 0, 25.798, 50.216, 73.252, 94.908, 115.182];

# coefficients in H0
  d_omega(1:5) = omega(2:6) - omega(1:5);

  # base wavelet
  tp = T;
  tc = 0.5*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(1) = envelope .*cos(d_omega(1)*t);
# state 2 to 3
  f(2) = envelope .*cos(d_omega(2)*t);
# state 3 to 4
  f(3) = envelope .*cos(d_omega(3)*t);

# period T/2 wavelets, centered at (0.25, 0.5, 0.75)*T
  tp = 0.5*T;

  tc = 0.25*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(4) = envelope .*cos(d_omega(1)*t);
# state 2 to 3
  f(5) = envelope .*cos(d_omega(2)*t);
# state 3 to 4
  f(6) = envelope .*cos(d_omega(3)*t);


  tc = 0.5*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(7) = envelope .*cos(d_omega(1)*t);
# state 2 to 3
  f(8) = envelope .*cos(d_omega(2)*t);
# state 3 to 4
  f(9) = envelope .*cos(d_omega(3)*t);

  tc = 0.75*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(10) = envelope .*cos(d_omega(1)*t);
# state 2 to 3
  f(11) = envelope .*cos(d_omega(2)*t);
# state 3 to 4
  f(12) = envelope .*cos(d_omega(3)*t);

# period T/4 wavelets, centered at (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)*T
  tp = 0.25*T;

  tc = 0.75*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(13) = envelope .*cos(d_omega(1)*t);
# state 2 to 3
  f(14) = envelope .*cos(d_omega(2)*t);
# state 3 to 4
  f(15) = envelope .*cos(d_omega(3)*t);
  
  tc = 0.875*T;
  tau = (t - tc)/tp;
  envelope = 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;
# from state 1 (ground) to state 2
  f(16) = envelope .*cos(d_omega(1)*t);
# state 2 to 3
  f(17) = envelope .*cos(d_omega(2)*t);
# state 3 to 4
  f(18) = envelope .*cos(d_omega(3)*t);
  
end
