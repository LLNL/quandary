%-*-octave-*--
%
% traceobjf1:
%
% USAGE:
% 
% [ineq_penalty] = eval_ineq_pen(pcof, par_0, par_1)
function [penalty] = eval_ineq_pen(pcof, par_0, par_1)
  D = size(pcof,1);
  N = size(pcof,2);
  scalef = 0.1;
  penalty = zeros(1,N);
  dp2 = (par_1 - par_0);
  for k=1:D
    penalty(1,:) = penalty(1,:) - ( log((pcof(k,:)-par_0)/dp2) + log((par_1 - pcof(k,:))/dp2) );
  end
  penalty = scalef*(penalty/D - 2*log(2));
# todo: need to normalize penalty such that it is still small when (par_1 - pcof) = ~ 0.05*(par_1 - par_0)
# pcof = par_1 - 0.05*(par_1 - par_0)
# (pcof - par_0)/(par_1 - par_0) = 0.95
# (par_1 - pcof)/(par_1 - par_0) = 0.05
end
