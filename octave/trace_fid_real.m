%-*-octave-*--
%
% USAGE:  f = trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega)
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
% fidelity2: | tr(vSol' *vTarget) | ^2
%
function  [fidelity2] = trace_fid_real(ur, vi, vTarget_r, vTarget_i, lab_frame, t, omega)
  N = size(vTarget_r,1);

  if (lab_frame)
				# verlet needs real arithmetic
    RotMat_c = diag([ cos(omega(1)*t), cos(omega(2)*t), cos(omega(3)*t), cos(omega(4)*t) ]);
    RotMat_s = diag([ sin(omega(1)*t), sin(omega(2)*t), sin(omega(3)*t), sin(omega(4)*t) ]);
    ua = RotMat_c *ur + RotMat_s * vi; # ur = + Re(u), vi = - Im(u)
    va = RotMat_s * ur - RotMat_c * vi;
  else
    ua = ur;
    va = -vi;
  end
  fidelity2 = (trace(ua' * vTarget_r + va' * vTarget_i)/N)^2 + (trace(ua' * vTarget_i - va' * vTarget_r)/N)^2;

end
