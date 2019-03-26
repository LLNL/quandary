%-*-octave-*--
%
% USAGE:  f =if30grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if30grad(t, param)
  D = size(param.pcof,1);
  if (D != 30)
    printf("ERROR: if30grad only works when D=30!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
