%-*-octave-*--
%
% USAGE:  f =if16grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if16grad(t, param)
  D = size(param.pcof,1);
  if (D != 16)
    printf("ERROR: if16grad only works when D=16!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
