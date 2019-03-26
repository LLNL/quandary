%-*-octave-*--
%
%
function  [da_r, da_i] = get_da_mat(t, amat, d_omega)

  dmat_r = diag([ cos(d_omega*(t)) ]);
  dmat_i = diag([ -sin(d_omega*(t)) ]);
  da_r = dmat_r * amat;
  da_i = dmat_i * amat;

end
