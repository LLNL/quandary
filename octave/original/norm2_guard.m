%-*-octave-*--
%
% USAGE:  f = norm2_guard(vr, vi, Nguard)
%
% INPUT:
% vr: vr = Re(uSol): real-valued solution matrix (Ntot x N)
% vi: vi = -Im(uSol): real-valued solution matrix (Ntot x N)
% Nguard: number of guard levels
%
% OUTPUT:
%
% sum_{k=N+1}^{Ntot} sum_{j=1}^N | uSol(k,j) | ^2
%
function  [f] = norm2_guard(vr, vi, Nguard)
  Ntot =size(vr,1);
  N = size(vr,2);

  f=0;
  if (Nguard>0)
    rguard = vr(N+1:N+Nguard,:); 
    iguard = vi(N+1:N+Nguard,:);
    f = sum(sumsq(rguard,2)) + sum(sumsq(iguard,2));

				# only consider the last guard level
    ## rguard = vr(Ntot,:); 
    ## iguard = vi(Ntot,:);
    ## f = sum(sumsq(rguard)) + sum(sumsq(iguard));
    
  end

end
