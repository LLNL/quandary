%-*-octave-*--
%
% USAGE:  f = trace_fid_cmplx(uSol, vTarget, lab_frame, t, omega)
%
% INPUT:
% uSol: Complex-valued solution matrix (NtotxN)
% vTarget:  Real-valued solution matrix (NtotxN)
% lab_frame: 0 or 1. If 1, rotate solution before evaluating fidelity
% t: time
% omega: real vector with eigen frequencies (N components)
%
% OUTPUT:
%
% fidelity2: | tr(vSol' *vTarget) | ^2
%
function  [fidelity2] = trace_fid_cmplx(uSol, vTarget, lab_frame, t, omega)
  N = size(vTarget,2);

  if (lab_frame)
    RotMat = diag([ exp(I*omega*t) ]);
    vSol = RotMat * uSol;
  else
    vSol = uSol;
  end

  fidelity2 = abs(trace(ctranspose(vSol) * vTarget)/N)^2;

end
