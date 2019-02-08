%-*-octave-*--
%
% USAGE:  f =if18grad(t, pcof)
%
% INPUT:
% t: time (real scalar)
% pcof: coefficient
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if18grad(t, pcof)
  D = size(pcof,1);
  if (D != 18)
    printf("ERROR: if18 only works when D=18!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
