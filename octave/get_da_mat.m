%-*-octave-*--
%
%
function  [da_r, da_i] = get_da_mat(t, amat, d_omega)

  dmat_r = diag([ cos(d_omega(1)*(t)), cos(d_omega(2)*(t)), cos(d_omega(3)*(t)), cos(d_omega(4)*(t)) ]);
  dmat_i = diag([ -sin(d_omega(1)*(t)), -sin(d_omega(2)*(t)), -sin(d_omega(3)*(t)), -sin(d_omega(4)*(t)) ]);
  da_r = dmat_r * amat;
  da_i = dmat_i * amat;

end
