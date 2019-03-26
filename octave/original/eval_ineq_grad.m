%-*-octave-*--
%
% traceobjf1:
%
% USAGE:
% 
% [grad] = eval_ineq_grad(pcof, par_0, par_1)
function [pen_grad] = eval_ineq_grad(pcof, par_0, par_1)
  D = size(pcof,1);
  N = size(pcof,2);
  scalef = 0.1;
  pen_grad = zeros(D,N);
  for k=1:D
    pen_grad(k,:) = scalef*(1.0./(par_1 - pcof(k,:)) - 1.0./(pcof(k,:)-par_0))/D;
  end

end
