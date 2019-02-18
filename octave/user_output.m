%-*-octave-*--
%
function [err] = user_output( x, optv, state )
  err = 0;
  printf("convergence report: state=%s, iter=%d, objf=%e\n", state, optv.iter, optv.fval);
  printf("x=[ %e", x(1));
  for q=2:length(x)
    printf(", %e", x(q));
  end
  printf("]\n");
end
