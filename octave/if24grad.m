%-*-octave-*--
%
% USAGE:  f =if24grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if24grad(t, param)
  D = size(param.pcof,1);
  if (D != 24)
    printf("ERROR: if24grad only works when D=24!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
