%-*-octave-*--
%
% tr_setup: setup data structure for optimization
%
% USAGE:
%  [x0, optop] = tr_setup( max_iter )
%
% INPUT:
% None
%
% OUTPUT:
% x0: column vector with the initial parameter guess
% optop: structure holding setting for optimizer
%
function [x0, optop] = tr_setup( max_iter )
  if nargin < 1
    max_iter = -1;
  end
  
  verbose=1;

				# 4 frequencies, xi=1
  x0 = [-1.2712e-03,   8.3987e-02,   1.0116e+00,   3.7495e-01]';

				# 4 frequencies, xi=0
  x0 =[-0.136577, 0.027567, 3.107615, -0.014116]';

  optop = optimset("GradObj", "on", "OutputFcn", @user_output);

  if (max_iter > 0)
    optop = optimset(optop, "MaxIter", max_iter);
  end

  if (verbose)
    printf("x0: "); x0
    printf("optop:\n"); optop
  end
end
