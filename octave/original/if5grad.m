%-*-octave-*--
%
% USAGE:  f =if5grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if5grad(t, param)
  D = size(param.pcof,1);
  if (D != 5)
    printf("ERROR: if5grad only works when D=5!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
