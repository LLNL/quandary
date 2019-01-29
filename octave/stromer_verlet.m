%-*-octave-*--
# Partitioned 2nd order RK method (Stromer-Verlet)
function [unew, vnew, tnew]=stromer_verlet(ur, vi, tfunc, t, dt, omega, K1, S1, Ident, separable, uforce, vforce)
  tf_0 = tfunc(t,omega);
  tf_1o2 = tfunc(t+0.5*dt,omega);
  tf_1 = tfunc(t+dt,omega);

  N = length(ur);
  				# RK stage variables
  kay1 = zeros(N,1);
  kay2 = zeros(N,1);
  ell1 = zeros(N,1);
  ell2 = zeros(N,1);

# forcing functions
  fu_1o2 = uforce(t+0.5*dt,omega);
  fv_0 = vforce(t,omega);
  fv_1 = vforce(t+dt,omega);

				#S1=0 -> fully explicit
  if (separable)
    ell1 = tf_0*K1*ur + fv_0;
    kay1 = - tf_1o2*K1*(vi+0.5*dt*ell1) + fu_1o2;
    kay2 = kay1;
    ell2 = tf_1*K1*(ur+0.5*dt*(kay1+kay2)) + fv_1;
  else
    rhs = tf_0*(K1*ur + S1*vi) + fv_0;
    ell1 = linsolve( Ident-0.5*dt*tf_0*S1, rhs );
    kay1 = tf_1o2 * (S1*ur - K1*(vi+0.5*dt*ell1) ) + fu_1o2;
    rhs = tf_1o2* (S1*(ur+0.5*dt*kay1) - K1*(vi+0.5*dt*ell1) ) + fu_1o2;
    kay2 = linsolve( Ident-0.5*dt*tf_1o2*S1, rhs );
    ell2 = tf_1* ( K1*(ur+0.5*dt*(kay1+kay2)) + S1*(vi+0.5*dt*ell1) ) + fv_1;
  end
				# update
  unew = ur + 0.5*dt*(kay1 + kay2);
  vnew = vi + 0.5*dt*(ell1 + ell2);
  tnew = t+dt;
end

