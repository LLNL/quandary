%-*-octave-*--
%
% USAGE:  f =if20grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if20grad(t, param)
  D = size(param.pcof,1);
  if (D != 20)
    printf("ERROR: if20grad only works when D=20!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
