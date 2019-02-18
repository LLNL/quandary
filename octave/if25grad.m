%-*-octave-*--
%
% USAGE:  f =if25grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if25grad(t, param)
  D = size(param.pcof,1);
  if (D != 25)
    printf("ERROR: if25grad only works when D=25!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
