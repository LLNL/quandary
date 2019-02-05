%-*-octave-*--
# Partitioned 2nd order RK method (Stromer-Verlet)
function [unew, vnew, tnew]=stromer_verlet_mat2(ur, vi, rfunc, ifunc, t, dt, pcof, H0, amat, Ident, d_omega, fu_0, fu_1, fv_0, fv_1)
  N = size(ur,1);
  				# RK stage variables
  kay1 = zeros(N,N);
  kay2 = zeros(N,N);
  ell1 = zeros(N,N);
  ell2 = zeros(N,N);

# forcing functions
#  fu_1o2 = 0.5*( uforce(t,pcof) + uforce(t+dt,pcof) );
  fu_1o2 = 0.5*( fu_0 + fu_1);
#  fv_0 = vforce(t,pcof);
#  fv_1 = vforce(t+dt,pcof);

	     # Evaluate sym and skew_sym matrices at the 3 time levels (lab frame)
  rf_0 = rfunc(t,pcof);
  rf_1o2 = rfunc(t+0.5*dt,pcof);
  rf_1 = rfunc(t+dt,pcof);

  if_0 = ifunc(t,pcof);
  if_1o2 = ifunc(t+0.5*dt,pcof);
  if_1 = ifunc(t+dt,pcof);
  
  # rotating frame
  dmat_r_0 = diag([ cos(d_omega(1)*(t)), cos(d_omega(2)*(t)), cos(d_omega(3)*(t)), cos(d_omega(4)*(t)) ]);
  dmat_i_0 = diag([ -sin(d_omega(1)*(t)), -sin(d_omega(2)*(t)), -sin(d_omega(3)*(t)), -sin(d_omega(4)*(t)) ]);

  dmat_r_1o2 = diag([ cos(d_omega(1)*(t+0.5*dt)), cos(d_omega(2)*(t+0.5*dt)), cos(d_omega(3)*(t+0.5*dt)), cos(d_omega(4)*(t+0.5*dt)) ]);
  dmat_i_1o2 = diag([ -sin(d_omega(1)*(t+0.5*dt)), -sin(d_omega(2)*(t+0.5*dt)), -sin(d_omega(3)*(t+0.5*dt)), -sin(d_omega(4)*(t+0.5*dt)) ]);

  dmat_r_1 = diag([ cos(d_omega(1)*(t+dt)), cos(d_omega(2)*(t+dt)), cos(d_omega(3)*(t+dt)), cos(d_omega(4)*(t+dt)) ]);
  dmat_i_1 = diag([ -sin(d_omega(1)*(t+dt)), -sin(d_omega(2)*(t+dt)), -sin(d_omega(3)*(t+dt)), -sin(d_omega(4)*(t+dt)) ]);


				# symmetric part
  K_0 =  rf_0.*(dmat_r_0 * amat +  amat' * dmat_r_0') - if_0.*(dmat_i_0 * amat + amat' * dmat_i_0');
  K_1o2 =  rf_1o2.*(dmat_r_1o2 * amat +  amat' * dmat_r_1o2') - if_1o2.*(dmat_i_1o2 * amat + amat' * dmat_i_1o2');
  K_1 =  rf_1.*(dmat_r_1 * amat +  amat' * dmat_r_1') - if_1.*(dmat_i_1 * amat + amat' * dmat_i_1');
				# skew-symmetric part
  S_0 =  if_0.*(dmat_r_0 * amat - amat' * dmat_r_0') + rf_0.*(dmat_i_0 * amat - amat' * dmat_i_0');
  S_1o2 =  if_1o2.*(dmat_r_1o2 * amat - amat' * dmat_r_1o2') + rf_1o2.*(dmat_i_1o2 * amat - amat' * dmat_i_1o2');
  S_1 =  if_1.*(dmat_r_1 * amat - amat' * dmat_r_1') + rf_1.*(dmat_i_1 * amat - amat' * dmat_i_1');

				#S1=0 -> fully explicit (never happens)
  rhs = (H0 + K_0)*ur + S_0*vi + fv_0;
  ell1 = linsolve( Ident-0.5*dt*S_0, rhs );
  kay1 =  S_1o2*ur - (H0 + K_1o2)*(vi+0.5*dt*ell1) + fu_1o2;
  rhs = S_1o2*(ur+0.5*dt*kay1) - (H0 + K_1o2)*(vi+0.5*dt*ell1) + fu_1o2;
  kay2 = linsolve( Ident-0.5*dt*S_1o2, rhs );
  ell2 = ( H0 + K_1)*(ur+0.5*dt*(kay1+kay2)) + S_1*(vi+0.5*dt*ell1) + fv_1;

				# update
  unew = ur + 0.5*dt*(kay1 + kay2);
  vnew = vi + 0.5*dt*(ell1 + ell2);
  tnew = t+dt;
end

