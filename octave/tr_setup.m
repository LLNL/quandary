%-*-octave-*--
%
% tr_setup: form the cell vector and initial guess for calling the 'sqp' nonlinear programming solver
%
% USAGE:
% 
% [x0, phi] = tr_setup(verbose)
%
% INPUT:
% None
%
% OUTPUT:
% x0: column vector with the solution of the 8-parameter control problem
% phi: cell-array with pointers to the objective and gradient functions
%
function [x0, phi] = tr_setup(verbose)

  if nargin < 1
    verbose=0;
  end

# initial guess for the 8-parameter control problem
  x0 = zeros(8,1);
  x0(1) = 0.26548;
  x0(2) = 0.26548;

  phi=cell(2,1);
  phi{1} = @traceobjf1;
  phi{2} = @tracegradient;

  if (verbose)
    printf("x0: "); x0
    printf("Using objective function traceobjf1 and gradient function tracegradient\n");
  end
end
