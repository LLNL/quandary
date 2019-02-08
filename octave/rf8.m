%-*-octave-*--
%
% USAGE:  f = rf8(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = rf8(t, pcof)
  D = size(pcof,1);
  if (D != 8)
    printf("ERROR: rf8 only works when D=8!\n");
    f=-999;
    return;
  end
  f = 0;

# Final time T
  global T;
# Frequency
  d1 = 24.64579437;
  
  # base wavelet
  tp = T;
  tc = 0.5*T;
  tau = (t - tc)/tp;
  f = pcof(1) * 64*(tau >= -0.5 & tau <= 0.5) .* (0.5 + tau).^3 .* (0.5 - tau).^3;

				# old
				# p  q
				# 1  1
				# 2  1
				# 3  2
				# 4  2
				# 5  3
				# 6  3
				# 7  4
				# 8  4

# period T/2 wavelets, centered at (0.25, 0.5, 0.75)*T
  tp = 0.5*T;

  tc = 0.25*T;
  tau = (t - tc)/tp;
  f = f + pcof(3) * 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

  tc = 0.5*T;
  tau = (t - tc)/tp;
  f = f + pcof(5) * 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

  tc = 0.75*T;
  tau = (t - tc)/tp;
  f = f + pcof(7) * 64*(tau >= -0.5 & tau <= 0.5).*(0.5 + tau).^3 .* (0.5 - tau).^3;

# from state 1 (ground) to state 2
  f = f.*cos(d1*t);

end
