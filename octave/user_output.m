%-*-octave-*--
%
function [err] = user_output( x, optv, state )
  err = 0;
  printf("fminunc report: state=%s, iter=%d, objf=%e\n", state, optv.iter, optv.fval);
end
