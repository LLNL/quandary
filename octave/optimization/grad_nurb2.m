%-*-octave-*--
%
% grad_nurb2: evaluate a quadratic nurb function
%
function  [g] = grad_nurb2(t, param)
  g = zeros(param.N_nurbs,1);

  dt_knot = param.dt_knot;
  t_knot = param.t_knot;
	# figure out where 't' is located in the knot array
  width = 3*dt_knot;

  k = max(3,ceil(t/dt_knot + 2)); # t_knot(k-1) < t <= t_knot(k), but t=0 needs to give k=3
  k = min(k, param.N_nurbs); # protect agains roundoff that sometimes makes t/dt > N_nurbs-2

				# 1st segment of nurb k-1
  tc = param.t_center(k);
  tau = (t - tc)/width;
  g(k) = (9/8 + 4.5*tau + 4.5*tau.^2);

				# 2nd segment of nurb k-2
  tc = param.t_center(k-1);
  tau = (t - tc)/width;
  g(k-1) = (0.75 - 9*tau.^2);
  
				# 3rd segment of nurb k-2
  tc = param.t_center(k-2);
  tau = (t - tc)/width;
  g(k-2) = (9/8 - 4.5*tau + 4.5*tau.^2);

end
