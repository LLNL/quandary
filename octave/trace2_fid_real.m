%-*-octave-*--
%
% USAGE:  trace2_fid_real = trace2_fid_cmplx(frc_r, frc_i, lambda_r, lambda_i)
%
% INPUT:
% frc_r: frc_r = Re(force): real part of forcing for phi (Ntot x N)
% frc_i: frc_i =  Im(force): imaginary part of forcing for phi (Ntot x N)
% lambda_r: Re(lambda): real part of adjoint solution (Ntot x N)
% lambda_i: Im(lambda): imaginary part of adjoint solution (Ntot x N)
%
% OUTPUT:
%
% fid_real: real(tr(vSol' *lambda))
%
function  [fid_real] = trace2_fid_real(frc_r, frc_i, lambda_r, lambda_i)

  Ntot = size(lambda_r,1);
  N = size(lambda_r,2);
  Nguard = Ntot - N;
  
  fid_real = trace(frc_r(1:N,:)' * lambda_r(1:N,:) + frc_i(1:N,:)' * lambda_i(1:N,:))/N;
# contributions from the guard levels is scaled differently
  if (Nguard > 0)
    fg_r = frc_r(N+1:N+Nguard,:);
    fg_i = frc_i(N+1:N+Nguard,:);
    lag_r = lambda_r(N+1:N+Nguard,:);
    lag_i = lambda_i(N+1:N+Nguard,:);
    fid_real = fid_real + trace(fg_r*lag_r') +  trace(fg_i*lag_i');
  end

end
