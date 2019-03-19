%-*-octave-*--
%
function [err] = user_output( x, optv, state )
  err = 0;
  printf("convergence report: state=%s, iter=%d, objf=%e\n", state, optv.iter, optv.fval);
  printf("x=[ %e", x(1));
  for q=2:min(10,length(x))
    printf(", %13.6e", x(q));
  end
  if (length(x)>10)
    printf(", (actual length = %d)\n", length(x));
  else
    printf("]\n");
  end
end
