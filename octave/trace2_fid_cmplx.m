%-*-octave-*--
%
% USAGE:  fid_cmplx = trace2_fid_cmplx(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega)
%
% INPUT:
% ur: ur = Re(uSol): real-valued solution matrix (NxN)
% vi: vi = -Im(uSol): real-valued solution matrix (NxN)
% vTarget_r: Re(vTarget): Real-valued solution matrix (NxN)
% vTarget_i: Im(vTarget):  Real-valued solution matrix (NxN)
% lab_frame: 0 or 1. If 0, vSol = uSol; If 1, rotate solution before evaluating the fidelity
% t: time
% omega: real vector with eigen frequencies (N components)
%
% OUTPUT:
%
% fid_cmplx: tr(vSol' *vTarget)
%
function  [fid_cmplx] = trace2_fid_cmplx(vr, vi, vTarget_r, vTarget_i, lab_frame, t, omega)
  N = size(vTarget_r,1);

  if (lab_frame)
				# verlet needs real arithmetic
    RotMat_c = diag([ cos(omega(1)*t), cos(omega(2)*t), cos(omega(3)*t), cos(omega(4)*t) ]);
    RotMat_s = diag([ sin(omega(1)*t), sin(omega(2)*t), sin(omega(3)*t), sin(omega(4)*t) ]);
    ua = RotMat_c * vr - RotMat_s * vi; # vr = + Re(u), vi = + Im(u)
    va = RotMat_s * vr + RotMat_c * vi;
  else
    ua = vr;
    va = vi;
  end
  fid_cmplx = trace(ua' * vTarget_r + va' * vTarget_i)/N + I*trace(ua' * vTarget_i - va' * vTarget_r)/N;

end
