%-*-octave-*--
# Forcing function the separable case
function [fv]=vforce2(t,omega)
  fv=zeros(2,1);
  phi1 = 0.25*(t - sin(omega*t)/omega);
  phi1dot = 0.5*(sin(0.5*omega*(t)))^2;
  fv(2) = -(tfunc2(t) - phi1dot) * cos(phi1);
end
