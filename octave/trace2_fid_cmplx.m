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
function  [fid_cmplx] = trace2_fid_cmplx(vr, vi, vTarget_r, vTarget_i, t, omega)

  N = size(vTarget_r,2);

  fid_cmplx = trace(vr' * vTarget_r + vi' * vTarget_i)/N + I*trace(vr' * vTarget_i - vi' * vTarget_r)/N;

end