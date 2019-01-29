%-*-octave-*--
# Time function for the separable case
function [tf]=tfunc1(t,omega)
  tf = 0.5*(sin(0.5*omega*(t)))^2;
end
