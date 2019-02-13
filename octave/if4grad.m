%-*-octave-*--
%
% USAGE:  f =if4grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if4grad(t, param)
  D = size(param.pcof,1);
  if (D != 4)
    printf("ERROR: if4grad only works when D=4!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
