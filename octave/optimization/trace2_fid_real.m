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

  fid_real = trace(frc_r' * lambda_r + frc_i' * lambda_i);

end
