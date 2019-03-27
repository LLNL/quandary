%-*-octave-*--
%
%
function  [f] = quad_spline(t, width, ctr)
  f = zeros(1,length(t));

  tp = width;
  tc = ctr;
  xi = (t - tc)/tp;
  f = (xi >= -0.5 & xi <= -1/6) .* (9/8 + 4.5*xi + 4.5*xi.^2);
  f = f + (xi >= -1/6 & xi <= 1/6) .* (0.75 - 9*xi.^2);
  f = f + (xi >= 1/6 & xi <= 0.5) .* (9/8 - 4.5*xi + 4.5*xi.^2);
  
end
