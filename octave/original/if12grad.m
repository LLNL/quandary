%-*-octave-*--
%
% USAGE:  f =irf12grad(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if12grad(t, pcof)
  D = size(pcof,1);
  if (D != 12)
    printf("ERROR: if12 only works when D=12!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
