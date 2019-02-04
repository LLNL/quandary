%-*-octave-*--
%
% plotunitary: plot traces of the solution of the schroedinger equation
%
% USAGE:
% 
% plotunitary()
%
% INPUT:
%
function usave = plotunitary(us, T, mode)
  if nargin < 1
    printf("ERROR: no solution provided\n");
    return;
  end

  if nargin < 2
    T=15;
  end

  if nargin < 3
    mode=0; # abs
  end

  nsteps = length(us(1,1,:));
  
  N1 = length(us(:,1,1));
  N2 = length(us(1,:,1));
#    printf("Data has dimensions %d x %d x %d\n", N1, N2, nsteps);

  if (N1 != N2)
    printf("ERROR: N1=%d and N2=%d must be equal!\n");
    return;
  end

  t = linspace(0,T,nsteps);

# one figure for the response of each basis vector
  for q=1:N2
    figure(q);
    if (mode==1)
      h=plot(t, real(us(1,q,:)), t, real(us(2,q,:)), t, real(us(3,q,:)), t, real(us(4,q,:)));
      legend("Re(u1)", "Re(u2)", "Re(u3)", "Re(u4)", "location", "east");
    else
      h=plot(t, abs(us(1,q,:)), t, abs(us(2,q,:)), t, abs(us(3,q,:)), t, abs(us(4,q,:)));
      legend("|u_1|", "|u_2|", "|u_3|", "|u_4|", "location", "east");
    end
    axis tight;
    set(h,"linewidth",2);
    tstr = sprintf("Response to initial data e_%1d", q);
    title(tstr);
    xlabel("Time");
  end
    
end
