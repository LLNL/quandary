%-*-octave-*--
%
% USAGE:  f = sc_real(vr, vi, wr, wi, Nguard)
%
% INPUT:
% vr: vr = Re(uSol): real-valued solution matrix (Ntot x N)
% vi: vi = -Im(uSol): real-valued solution matrix (Ntot x N)
% wr: wr = Re(Phi): real-valued solution matrix (Ntot x N)
% wi: wi = -Im(Phi): real-valued solution matrix (Ntot x N)
% Nguard: number of guard levels
%
% OUTPUT:
%
% sum_{k=N+1}^{Ntot} sum_{j=1}^N vr(k,j)*wr(k,j) + vi(k,j)*wi(k,j)
%
function  [f] = sc_real(vr, vi, wr, wi, Nguard)
  Ntot =size(vr,1);
  N = size(vr,2);

  f=0;
  if (Nguard>0)
    vrguard = vr(N+1:N+Nguard,:);
    viguard = vi(N+1:N+Nguard,:);
    wrguard = wr(N+1:N+Nguard,:);
    wiguard = wi(N+1:N+Nguard,:);
    
    f = trace(vrguard*wrguard') +  trace(viguard*wiguard');
  end

end
