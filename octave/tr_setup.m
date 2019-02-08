%-*-octave-*--
%
% tr_setup: form the cell vector and initial guess for calling the 'sqp' nonlinear programming solver
%               also assign the global variable T, which holds the duration of the control function and the simulation
%
% SIDE EFFECT: this routine uses the "clear all" command to allow the global variable T to be changed.
%
% USAGE:
% 
% [x0, phi] = tr_setup()
%
% INPUT:
% None
%
% OUTPUT:
% x0: column vector with the solution of the 8-parameter control problem
% phi: cell-array with pointers to the objective and gradient functions
%
function [x0, phi] = tr_setup()
  
  clear all;
  # set duration of control function
  global T=20;

  verbose=1;

# initial guess for the 6-parameter control problem
#  x0 = zeros(6,1);
  # 2-param optima
#  x0(1) = 0.26548;
#  x0(2) = 0.26548;
# 8 param opt:
  ## x0(1:2) =  0.4385710;
  ## x0(3:4) =  0.0016887;
  ## x0(5:6) = -0.0383177;
  ## x0(7:8) = -0.3891266;

				# 24 param opt (T=15):
##  x0 = zeros(24,1);
  ## x0 = [ 0.983838; 0.277051; -0.288670; 0.424397; -0.039568; -0.423635; -0.288176; -0.193112; -0.131186; -0.056086; 0.161684; 0.034695;
  ## 	 -0.021766; -0.217401; -0.127394; 0.358467; 0.398741;  0.222429; -0.424639; -0.393889; -0.266185; -0.098312;  0.216814; -0.136686];

				# 24 param optima (T=20):
  ## x0 =[ 1.027135e+00, 3.484092e-01, 1.021566e+00, 1.301384e-01, 9.463473e-01, 3.174267e-01, -3.038868e-01, -1.685488e-01, -2.125763e-01, -7.500755e-02, -5.895253e-02, 1.296375e-01, -4.142287e-01, 2.010823e-01, -5.796396e-01, 1.804119e-01, -1.064056e+00, -5.741716e-01, 9.878458e-02, -5.437661e-01, 1.605615e-01, -6.659033e-01, 3.058232e-01, -3.263869e-01 ]';

  ## x0 =[ 1.375544e+00, 0.000000e+00, 1.151704e+00, 0.000000e+00, 1.263774e+00, 0.000000e+00, -4.724356e-01, 0.000000e+00, -2.125763e-01, -7.500755e-02, -5.895253e-02, 1.296375e-01, -4.142287e-01, 2.010823e-01, -5.796396e-01, 1.804119e-01, -1.064056e+00, -5.741716e-01, 9.878458e-02, -5.437661e-01, 1.605615e-01, -6.659033e-01, 3.058232e-01, -3.263869e-01 ]';

				# 12 parameter optimia (all real) T=20, obj=1.03e-2
  ## x0 = [1.252313922528511e+00,
  ## 	   1.132658657203102e+00,
  ## 	   1.193828504090577e+00,
  ## 	   -4.088820659527235e-01,
  ## 	   -2.947845371034721e-01,
  ## 	   1.059197013478711e-01,
  ## 	   -7.017460199667226e-02,
  ## 	   -3.817933834658930e-01,
  ## 	   -1.560434946604147e+00,
  ## 	   -3.804175469131835e-01,
  ## 	   -4.841119766415518e-01,
  ## 	   2.258260197267531e-02]';

  #18 real parameters, T=20
  x0 = [1.252313922528511e+00,
	   1.132658657203102e+00,
	   1.193828504090577e+00,
	   -4.088820659527235e-01,
	   -2.947845371034721e-01,
	   1.059197013478711e-01,
	   -7.017460199667226e-02,
	   -3.817933834658930e-01,
	   -1.560434946604147e+00,
	   -3.804175469131835e-01,
	   -4.841119766415518e-01,
	   2.258260197267531e-02,
	   0,
	   0,
	   0,
	   0,
	   0,
	   0];


  phi=cell(2,1);
  phi{1} = @traceobjf1;
  phi{2} = @tracegradient;

  if (verbose)
    printf("x0: "); x0
    printf("Using objective function traceobjf1 and gradient function tracegradient\n");
  end
end
