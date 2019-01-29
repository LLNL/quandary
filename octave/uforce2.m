%-*-octave-*--
# Time function for the separable case
function [fu]=uforce2(t,omega)
  fu=zeros(2,1);
  phi1 = 0.25*(t - sin(omega*t)/omega);
  phi1dot = 0.5*(sin(0.5*omega*(t)))^2;
  fu(1) = (tfunc2(t) - phi1dot) * sin(phi1);
end
