%-*-octave-*--
# Time function for the twilight case
function [tf]=tfunc2(t,omega)
  T = 5*pi;
  tf = 4/T^2 *t*(T-t);
end
