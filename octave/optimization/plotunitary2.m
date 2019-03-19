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
function usave = plotunitary2(us, T, mode)
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
  
  Ntot = length(us(:,1,1));
  N = length(us(1,:,1));
#    printf("Data has dimensions %d x %d x %d\n", N1, N2, nsteps);

  if (Ntot != N)
    printf("INFO plotunitary2: Ntot=%d and N=%d are not equal!\n", Ntot, N);
  end

  t = linspace(0,T,nsteps);

# one figure for the response of each basis vector
  for q=1:N
    figure(q);
    if (mode==1) # real part
      if Ntot == 6
	h=plot(t, real(us(1,q,:)), t, real(us(2,q,:)), t, real(us(3,q,:)), t, real(us(4,q,:)), t, real(us(5,q,:)), t, real(us(6,q,:)));
	legend("Re(u1)", "Re(u2)", "Re(u3)", "Re(u4)", "Re(u5)", "Re(u6)", "location", "east");
      elseif Ntot == 7
	h=plot(t, real(us(1,q,:)), t, real(us(2,q,:)), t, real(us(3,q,:)), t, real(us(4,q,:)), t, real(us(5,q,:)), ...
	       t, real(us(6,q,:)), t, real(us(7,q,:)));
	legend("Re(u1)", "Re(u2)", "Re(u3)", "Re(u4)", "Re(u5)", "Re(u6)", "Re(u7)", "location", "east");
      elseif Ntot == 4
	h=plot(t, real(us(1,q,:)), t, real(us(2,q,:)), t, real(us(3,q,:)), t, real(us(4,q,:)) );
	legend("Re(u1)", "Re(u2)", "Re(u3)", "Re(u4)", "location", "east");
      end
    else #abs
      if Ntot == 6
	h=plot(t, abs(us(1,q,:)), t, abs(us(2,q,:)), t, abs(us(3,q,:)), t, abs(us(4,q,:)), t, abs(us(5,q,:)), t, abs(us(6,q,:)));
	legend("|u_1|", "|u_2|", "|u_3|", "|u_4|", "|u_5|", "|u_6|", "location", "east");
      elseif Ntot == 4
	h=plot(t, abs(us(1,q,:)), t, abs(us(2,q,:)), t, abs(us(3,q,:)), t, abs(us(4,q,:)) );
	legend("|u_1|", "|u_2|", "|u_3|", "|u_4|", "location", "east");
      elseif Ntot == 7
	h=plot(t, abs(us(1,q,:)), t, abs(us(2,q,:)), t, abs(us(3,q,:)), t, abs(us(4,q,:)), t, abs(us(5,q,:)), ...
	       t, abs(us(6,q,:)), t, abs(us(7,q,:)));
	legend("|u_1|", "|u_2|", "|u_3|", "|u_4|", "|u_5|", "|u_6|", "|u_7|", "location", "east");
      end
    end
    axis tight;
    set(h,"linewidth",2);
    tstr = sprintf("Response to initial data e_%1d", q);
    title(tstr);
    xlabel("Time");
  end
    
end
