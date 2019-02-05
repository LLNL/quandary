%-*-octave-*--
%
% tracegradient: solve a model problem from quantum control theory
%
% USAGE:
% 
% [objF, uFinal] = tracegradient(pcof, dp, order, verbose)
%
% INPUT:
% pcof(D,1): amplitudes of the control functions as a D x 1 column vector, D=size(pcof,1)
% verbose: 0: quite mode, 1: verbose
% order: order of accuracy: 2, 4, or 6.
%
% OUTPUT:
% objF: trace norm of gate infidelity cost functional
% uFinal_r: Real part of state vector at t=T
% uFinal_i: Imaginary part of state vector at t=T
%
function [dfdp] = tracegradient(pcof0, dp, order, verbose)

  if (nargin<4)
    verbose=0;
  end

  f0 = traceobjf1(pcof0, order);

  pcof1 = pcof0 + [dp; 0]; # perturb coefficient  number 1

  f1 = traceobjf1(pcof1, order);

# divided difference approximation
  dfdp = (f1-f0)/dp;

  if (verbose)
    printf("pcof0: ")
    for q=1:length(pcof0)
      printf(" %e", pcof0(q));
    end
    printf("\n");
    printf("dp1 = %e, f1 = %e, f0 = %e\n", dp, f1, f0);
    printf("(f1-f0)/dp = %e\n", dfdp);
  end
end
