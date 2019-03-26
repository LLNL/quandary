%-*-octave-*--
%
% tr_set_global: assign global variables used by traceobjgrad
%
%
% USAGE:
%  tr_set_global( )
%
% INPUT:
% None
%
% OUTPUT:
% None
%
% SIDE EFFECT: this routine uses the "clear all" command to allow the global variables T and xi to be changed.
function tr_set_global( )
  
  clear all;
  # set duration of control function
  global T=20;
# set coefficient for penalizing forbidden states
  global xi=0.0;

end
