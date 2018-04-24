%-*-octave-*--
%
% setup: form the cell vector and initial guess for calling the 'sqp' nonlinear programming solver
%
% USAGE:
% 
% [phi, a0] = objective()
%
% INPUT:
% None
%
% OUTPUT:
% phi: cell-array with pointers to the objective and gradient functions
% a0: column vector with the solution of the 8-parameter control problem
%
function [phi a0] = setup(verbose)

  if nargin < 1
    verbose=0;
  end

# solution to the 8-parameter control problem
  a0 = [ 0.4077076; -0.0068228; 0.0487782; 0.0871806; -0.0616958; 0.0147906; 0.0769747; -0.0732247];

  phi=cell(2,1);
  phi{1} = @objective;
  phi{2} = @gradient;
  
end
