%-*-octave-*--
%
% traceobjf1: evaluate the objective function using the trace norm
%
% USAGE:
% 
% nurbstest(N_spline)
%
% INPUT:
% N_spline: number of splines >=3

% OUTPUT:
%
function nurbstest(N_nurbs)

  T=120;# final time
#  T=5;# final time

# discretize time
  dt = 1/20; 
  N_steps = ceil(T/dt);
  dt = T/N_steps;
  printf("Final time = %e, number of time steps = %d, time step = %e\n", T, N_steps, dt);

  N_knots = N_nurbs+1;
  dt_knot = T/(N_nurbs-2);
  width = 3*dt_knot;
  t_center = dt_knot*([1:N_nurbs] - 1.5);
  t_knot = dt_knot*( [1:N_knots] - 2 );

  pcof = zeros(N_nurbs,1);
  pcof = ones(N_nurbs,1);
  ## for q=1:2:N_nurbs
  ##   pcof(q) = -1;
  ## end
  pcof = 2*rand(N_nurbs,1) - 1;
  ## pcof(1:2) = 0;
  ## pcof(N_nurbs) = 0;
  ## pcof(N_nurbs - 1) = 0;

 # param: struct for passing parameters to the time function
 # T is a real positive scalar
 # N_nurbs: number of nurbs ( positive integer ) 
 # N_knots: number of knots (N_nurbs+3)
 # t_center: Center time for each nurb (1 x N_nurbs) array of real
 # t_knot: Time corresponding to each knot (1 x N_knots) array of real
 # dt_knot: Spacing in t_knot array
 # pcof: Nurbs coefficients (N_nurbs x 1) array of real
  param = struct("T", T, "N_nurbs", N_nurbs, "t_knot", t_knot, "dt_knot", dt_knot, "t_center", t_center, "pcof", pcof);

# evaluate the nurb on a grid
#  td = linspace(width, T-width, N_steps);
  td = linspace(0, T, N_steps);
#  td=0.5*T;
  ctrl = nurb2(td, param);
  printf("size(ctrl)= (%d %d), max(ctrl)=%e\n", size(ctrl), max(ctrl) );
  
  figure(1);
  plot(td, ctrl, "b-");
  axis([0, T, -1.1, 1.1]);
  xlabel("Time [s]");
  
				# FFT on the control function
  df = 1/T;
  Nf = N_steps;
  freq=[ -(ceil((Nf-1)/2):-1:1), 0, (1:floor((Nf-1)/2)) ] * df; # In Hz
  om = 2*pi*freq; % angular frequency in rad/sec

  fctrl = fftshift(fft(ifftshift(ctrl)));
  fctrl = fctrl/Nf;

  printf("size(fctrl)= (%d %d), max(abs(fctrl))=%e\n", size(ctrl), max(abs(fctrl)) );
  
  figure(2);
  semilogy(om, abs(fctrl)+1e-18, "ro"); #1e-18 is to stop the plot function from complaining
  axis([0, 10, 1e-5, max(abs(fctrl))]);
  npar = length(pcof);
  tstr = sprintf("FFT(drive), %d nurbs", N_nurbs);
  title(tstr);
  xlabel("Omega [rad/s]");

  # test the grad_nurbs2 function
  t = 12.34;
  k = max(3,ceil(t/param.dt_knot + 2)); # t_knot(k-1) < t <= t_knot(k), but t=0 needs to give k=3
  
  printf("Testing grad_nurb2, t=%e, k=%d, gradient:\n", t, k);
  g=grad_nurb2(t, param);

				# FD test
  dp = 1e-5;
  f0=nurb2(t, param);
  q=k-2;
  param.pcof(q)= param.pcof(q)+dp;
  f1=nurb2(t, param);
  
  printf("g(%d)=%e, (f1-f0)/dp = %e\n", q, g(q), (f1-f0)/dp);

  
end
