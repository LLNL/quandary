%-*-octave-*--
%
% tr_setup: setup data structure for optimization
%
% USAGE:
%  [x0, optop] = tr_setup( verbose, max_iter )
%
% INPUT:
% verbose: 1 for verbose mode (0 = default)
% max_iter: Max number of iterations encoded in the optop structure (Positive integer)
% None
%
% OUTPUT:
% x0: column vector with the initial parameter guess
% optop: structure holding runtime info for the fminunc optimizer
%
function [x0, optop] = tr_setup( verbose, max_iter )
  if nargin < 2
    max_iter = -1;
  end
  
  if nargin < 1
    verbose=0;
  end
  
  optop = optimset("GradObj", "on", "OutputFcn", @user_output);

# test for b-splines
  x0 = zeros(150,1);
  x0(5) = 0.05;
  x0(6) = -0.05;
  
  if (max_iter > 0)
    optop = optimset(optop, "MaxIter", max_iter);
  end

  if (verbose)
    printf("x0: "); x0
    printf("optop:\n"); optop
  end
end
