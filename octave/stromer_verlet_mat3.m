%-*-octave-*--
# Partitioned 2nd order RK method (Stromer-Verlet)
function [unew, vnew, tnew]=stromer_verlet_mat3(ur, vi, rfunc, ifunc, t, dt, param, H0, amat, Ident, d_omega, fu_0, fu_1, fv_0, fv_1)
  Ntot = size(ur,1);
  N = size(ur,2);
  				# RK stage variables
  kay1 = zeros(Ntot,N);
  kay2 = zeros(Ntot,N);
  ell1 = zeros(Ntot,N);
  ell2 = zeros(Ntot,N);

# forcing functions
#  fu_1o2 = 0.5*( uforce(t,param) + uforce(t+dt,param) );
  fu_1o2 = 0.5*( fu_0 + fu_1);
#  fv_0 = vforce(t,param);
#  fv_1 = vforce(t+dt,param);

	     # Evaluate sym and skew_sym matrices at the 3 time levels (lab frame)
  rf_0 = rfunc(t,param);
  rf_1o2 = rfunc(t+0.5*dt,param);
  rf_1 = rfunc(t+dt,param);

  if_0 = ifunc(t,param);
  if_1o2 = ifunc(t+0.5*dt,param);
  if_1 = ifunc(t+dt,param);
  
  # rotating frame
  dmat_r_0 = diag([ cos(d_omega*(t)) ]);
  dmat_i_0 = diag([ -sin(d_omega*(t)) ]);

  dmat_r_1o2 = diag([ cos(d_omega*(t+0.5*dt)) ]);
  dmat_i_1o2 = diag([ -sin(d_omega*(t+0.5*dt)) ]);

  dmat_r_1 = diag([ cos(d_omega*(t+dt)) ]);
  dmat_i_1 = diag([ -sin(d_omega*(t+dt)) ]);


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

