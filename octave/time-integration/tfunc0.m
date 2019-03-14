%-*-octave-*--
# Time function for the non-separable test case
function [tf]=tfunc0(t,omega)
  tf = 0.25*(1-sin(omega*t));
end
