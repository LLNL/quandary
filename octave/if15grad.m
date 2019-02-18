%-*-octave-*--
%
% USAGE:  f =if15grad(t, param)
%
% INPUT:
% t: time (real scalar)
% param: struct of coefficients
%
% OUTPUT:
%
% f: real part of time function at time t
%
function  [f] = if15grad(t, param)
  D = size(param.pcof,1);
  if (D != 15)
    printf("ERROR: if15grad only works when D=15!\n");
    f=-999;
    return;
  end
  f = zeros(D,1);
  
end
