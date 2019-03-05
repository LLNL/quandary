%-*-octave-*--
%
% zero_grad: gradient of the zero function
%
function  [g] = zero_grad(t, param)
  g = zeros(param.N_nurbs,1);
end
