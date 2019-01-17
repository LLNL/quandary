%-*-octave-*--
# Partitioned 2nd order RK method (Stromer-Verlet)
function [unew, vnew, tnew]=stromer_verlet(ur, vi, t, dt, omega, K1, S1, Ident, separable)
  if (separable)
    tf_0 = 0.5*(sin(0.5*omega*(t)))^2;
    tf_1o2=0.5*(sin(0.5*omega*(t+0.5*dt)))^2;
    tf_1 =0.5*(sin(0.5*omega*(t+dt)))^2;
  else
    tf_0 = 0.25*(1-sin(omega*(t)));
    tf_1o2 = 0.25*(1-sin(omega*(t+0.5*dt)));
    tf_1 = 0.25*(1-sin(omega*(t+dt)));
  end      

  N = length(ur);
  				# RK stage variables
  kay1 = zeros(N,1);
  kay2 = zeros(N,1);
  ell1 = zeros(N,1);
  ell2 = zeros(N,1);

				#S1=0 -> fully explicit
  if (separable)
    ell1 = tf_0*K1*ur;
    kay1 = - tf_1o2*K1*(vi+0.5*dt*ell1);
    kay2 = kay1;
    ell2 = tf_1*K1*(ur+0.5*dt*(kay1+kay2));
  else
    rhs = tf_0*(K1*ur + S1*vi);
    ell1 = linsolve( Ident-0.5*dt*tf_0*S1, rhs );
    kay1 = tf_1o2 * (S1*ur - K1*(vi+0.5*dt*ell1) );
    rhs = tf_1o2* (S1*(ur+0.5*dt*kay1) - K1*(vi+0.5*dt*ell1) );
    kay2 = linsolve( Ident-0.5*dt*tf_1o2*S1, rhs );
    ell2 = tf_1* ( K1*(ur+0.5*dt*(kay1+kay2)) + S1*(vi+0.5*dt*ell1) );
  end
				# update
  unew = ur + 0.5*dt*(kay1 + kay2);
  vnew = vi + 0.5*dt*(ell1 + ell2);
  tnew = t+dt;
end

