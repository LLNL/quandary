%-*-octave-*--
%
% USAGE:  f = tf1(t, acof)
%
% INPUT:
% t: time (real scalar)
% acof: coefficient
%
% OUTPUT:
%
% f: time function at time t
%
function  [f] = tf1(t, acof)

# Final time T
  global T;
# Frequency
  d1 = 24.64579437;
  
  ## if nargin < 3
  ##   verbose=0;
  ## end

  ## if (verbose==1)
  ##   printf("Final time = %e, angular frequency = %e, acof = %e\n", ...
  ## 	   T, d1, acof);
  ## end

  p = 1;
  q = floor((p-1)/2) + 1;
  if (q==1)
    tp = T;
    t0 = 0.5*T;
  elseif (q > 1 & q <=4)
    tp = 0.5*T;
    t0 = 0.25*(q-1)*T;
  elseif (q > 4 & q <= 11)
    tp = 0.25*T;
    t0 = 0.125*(q-4)*T;
  elseif (q > 11 & q <= 26)
    tp = 0.125*T;
    t0 = 0.0625*(q-11)*T;
  end
  
  tau = (t - t0)/tp;
  mask = (tau >= -0.5 & tau <= 0.5);

  if (mod(p,2) == 1)
#      printf("Assigning p=%d, 2*q-1=%d\n", p, 2*q-1);
    f = acof * 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3 .*cos(d1*t);
  else
#      printf("Assigning p=%d, 2*q=%d\n", p, 2*q);
    f = 64*mask.*(0.5 + tau).^3 .* (0.5 - tau).^3 .*sin(d1*t);
  end

end
