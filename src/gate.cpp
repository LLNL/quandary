#include "gate.hpp"

Gate::Gate(){
  dim_ess = 0;
  dim_rho = 0;
}

Gate::Gate(std::vector<int> nlevels_, std::vector<int> nessential_, double time_, std::vector<double> gate_rot_freq_){

  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);

  nessential = nessential_;
  nlevels = nlevels_;
  final_time = time_;
  gate_rot_freq = gate_rot_freq_;
  for (int i=0; i<gate_rot_freq.size(); i++){
    gate_rot_freq[i] *= 2.*M_PI;
  }

  /* Dimension of gate = \prod_j nessential_j */
  dim_ess = 1;
  for (int i=0; i<nessential.size(); i++) {
    dim_ess *= nessential[i];
  }

  /* Dimension of system matrix rho */
  dim_rho = 1;
  for (int i=0; i<nlevels.size(); i++) {
    dim_rho *= nlevels[i];
  }

  /* Allocate input Gate in essential level dimension, sequential matrix (real and imaginary parts), copied on all processors */
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &V_re);
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &V_im);
  MatSetUp(V_re);
  MatSetUp(V_im);
  MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);

  /* Allocate vectorized Gate in full dimensions G = VxV, where V is the full-dimension gate (inserting zero rows and colums for all non-essential levels) */ 
  // parallel matrix, essential levels dimension TODO: Preallocate!
  MatCreate(PETSC_COMM_WORLD, &VxV_re);
  MatCreate(PETSC_COMM_WORLD, &VxV_im);
  MatSetSizes(VxV_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetSizes(VxV_im, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetUp(VxV_re);
  MatSetUp(VxV_im);
  MatAssemblyBegin(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(VxV_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_im, MAT_FINAL_ASSEMBLY);

  /* Allocate auxiliare vectors */
  MatCreateVecs(VxV_re, &x, NULL);


  /* Create vector strides for accessing real and imaginary part of co-located state */
  int ilow, iupp;
  MatGetOwnershipRange(VxV_re, &ilow, &iupp);
  int dimis = iupp - ilow;
  ISCreateStride(PETSC_COMM_WORLD, dimis, 2*ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, 2*ilow+1, 2, &isv);
 
}

Gate::~Gate(){
  if (dim_rho == 0) return;
  MatDestroy(&VxV_re);
  MatDestroy(&VxV_im);
  MatDestroy(&V_re);
  MatDestroy(&V_im);
  VecDestroy(&x);
  ISDestroy(&isu);
  ISDestroy(&isv);
}

void Gate::assembleGate(){

  /* Rorate the gate to rotational frame. */
  const PetscScalar* vals_vre, *vals_vim;
  PetscScalar *out_re, *out_im;
  int *cols;
  PetscMalloc1(dim_ess, &out_re); 
  PetscMalloc1(dim_ess, &out_im); 
  PetscMalloc1(dim_ess, &cols); 
  // get the frequency of the diagonal scaling e^{iwt} for each row rotation matrix R=R1\otimes R2\otimes...
  for (int row=0; row<dim_ess; row++){
    int r = row;
    double freq = 0.0;
    for (int iosc=0; iosc<nlevels.size(); iosc++){
      // compute dimension of essential levels of all following subsystems 
      int dim_post = 1;
      for (int josc=iosc+1; josc<nlevels.size();josc++) {
        dim_post *= nessential[josc];
      }
      // compute the frequency 
      int rk = (int) r / dim_post;
      freq = freq + rk * gate_rot_freq[iosc];
      r = r % dim_post;
    }
    double ra = cos(freq*final_time);
    double rb = sin(freq*final_time);
    /* Get row in V that is to be scaled by the rotation */
    MatGetRow(V_re, row, NULL, NULL, &vals_vre);  // V_re, V_im is stored dense , so ncols = dim_ess!
    MatGetRow(V_im, row, NULL, NULL, &vals_vim);
    // Compute the rotated real and imaginary part
    for (int c=0; c<dim_ess; c++){        
      out_re[c] = ra * vals_vre[c] - rb * vals_vim[c];
      out_im[c] = ra * vals_vim[c] + rb * vals_vre[c];
      cols[c] = c;
    }
    MatRestoreRow(V_re, row, NULL, NULL, &vals_vre);
    MatRestoreRow(V_im, row, NULL, NULL, &vals_vim);
    // Insert the new values
    MatSetValues(V_re, 1, &row, dim_ess, cols, out_re, INSERT_VALUES);
    MatSetValues(V_im, 1, &row, dim_ess, cols, out_im, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);
  }
  // clean up
  PetscFree(out_re);
  PetscFree(out_im);
  PetscFree(cols);

#ifdef SANITY_CHECK
  bool isunitary = isUnitary(V_re, V_im);
  if (!isunitary) {
    printf("ERROR: Rotated Gate is not unitary!\n");
    exit(1);
  } 
  else printf("Rotated Gate is unitary.\n");
#endif


  /* Assemble vectorized gate G=V\kron V where V = PV_eP^T for essential dimension gate V_e (user input) and projection P lifting V_e to the full dimension by inserting identity blocks for non-essential levels. */
  // Each element in V\kron V is a product V(i,j)*V(r,c), for rows and columns i,j,r,c!
  int ilow, iupp;
  MatGetOwnershipRange(VxV_re, &ilow, &iupp);
  double val;
  double vre_ij, vim_ij;
  double vre_rc, vim_rc;
  // iterate over rows of V_e (essential dimension gate)
  for (int row_f=0;row_f<dim_rho; row_f++) {
    if (isEssential(row_f, nlevels, nessential)) { // place \bar v_xx*V_f blocks for all cols in V_e[row_e]
      int row_e = mapFullToEss(row_f, nlevels, nessential);
      assert(row_f == mapEssToFull(row_e, nlevels, nessential));
      // iterate over columns in this row_e
      for (int col_e=0; col_e<dim_ess; col_e++) {
        vre_ij = 0.0; vim_ij = 0.0;
        MatGetValues(V_re, 1, &row_e, 1, &col_e, &vre_ij);
        MatGetValues(V_im, 1, &row_e, 1, &col_e, &vim_ij);
        // for all nonzeros in this row, place block \bar Ve_{i,j} * (V_f) at starting position G[a,b]
        if (fabs(vre_ij) > 1e-14 || fabs(vim_ij) > 1e-14 ) {
          int a = row_f * dim_rho;
          int b = mapEssToFull(col_e, nlevels, nessential) * dim_rho;
          // iterate over rows in V_f
          for (int r=0; r<dim_rho; r++) {
            int rowout = a + r;  // row in G
            if (ilow <= rowout && rowout < iupp) {
              if (isEssential(r, nlevels, nessential)){ // place ve_ij*ve_rc at G[a+r, b+map(ce)]
                int re = mapFullToEss(r, nlevels, nessential);
                for (int ce=0; ce<dim_ess; ce++) {
                  int colout = b + mapEssToFull(ce, nlevels, nessential); // column in G
                  vre_rc = 0.0; vim_rc = 0.0;
                  MatGetValues(V_re, 1, &re, 1, &ce, &vre_rc);
                  MatGetValues(V_im, 1, &re, 1, &ce, &vim_rc);
                  val = vre_ij*vre_rc + vim_ij*vim_rc;
                  if (fabs(val) > 1e-14) MatSetValue(VxV_re, rowout, colout, val, INSERT_VALUES);
                  val = vre_ij*vim_rc - vim_ij*vre_rc;
                  if (fabs(val) > 1e-14) MatSetValue(VxV_im, rowout, colout, val, INSERT_VALUES);
                }  
              } else { // place ve_ij*1.0 at G[a+row, a+row]
              int colout = b + r;
                  val = vre_ij;
                  if (fabs(val) > 1e-14) MatSetValue(VxV_re, rowout, colout, val, INSERT_VALUES);
                  val = vim_rc;
                  if (fabs(val) > 1e-14) MatSetValue(VxV_im, rowout, colout, val, INSERT_VALUES);
              }              
            }
          }
        }
      }
    } else { // place Vf block starting at G[a,a], a=row_f * N
      int a = row_f * dim_rho;
      // iterate over rows in V_f
      for (int r=0; r<dim_rho; r++) {
        int rowout = a + r;  // row in G
        if (ilow <= rowout && rowout < iupp) {
          if (isEssential(r, nlevels, nessential)){ // place ve_rc at G[a+r, a+map(ce)]
            int re = mapFullToEss(r, nlevels, nessential);
            for (int ce=0; ce<dim_ess; ce++) {
              int colout = a + mapEssToFull(ce, nlevels, nessential); // column in G
              vre_rc = 0.0; vim_rc = 0.0;
              MatGetValues(V_re, 1, &re, 1, &ce, &vre_rc);
              MatGetValues(V_im, 1, &re, 1, &ce, &vim_rc);
              val = vre_rc;
              if (fabs(val) > 1e-14) MatSetValue(VxV_re, rowout, colout, val, INSERT_VALUES);
              val = vim_rc;
              if (fabs(val) > 1e-14) MatSetValue(VxV_im, rowout, colout, val, INSERT_VALUES);
            }  
          } else { // place 1.0 at G[a+r, a+r]
              int colout = a + r;
              val = 1.0;
              if (fabs(val) > 1e-14) MatSetValue(VxV_re, rowout, colout, val, INSERT_VALUES);
          }              
        }
      }
    }
  }

  MatAssemblyBegin(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(VxV_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_im, MAT_FINAL_ASSEMBLY);

}


void Gate::compare_frobenius(const Vec finalstate, const Vec rho0, double& frob){
  frob = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim_rho == 0) {
    return;
  }

  /* Get real and imag part of final state and initial state */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* Add real part of frobenius norm || u - VxV_re*u0 + VxV_im*v0 ||^2 */
  double norm;
  MatMult(VxV_re, u0, x);            // x = VxV_re*u0
  VecAYPX(x, -1.0, ufinal);       // x = ufinal - VxV_re*u0 
  MatMultAdd(VxV_im, v0, x, x);      // x = ufinal - VxV_re*u0 + VxV_im*v0
  VecNorm(x, NORM_2, &norm);
  frob = pow(norm,2.0);           // frob = || x ||^2

  /* Add imaginary part of frobenius norm || v - VxV_re*v0 - VxV_im*u0 ||^2 */
  MatMult(VxV_re, v0, x);         // x = VxV_re*v0
  MatMultAdd(VxV_im, u0, x, x);   // x = VxV_re*v0 + VxV_im*u0
  VecAYPX(x, -1.0, vfinal);     // x = vfinal - (VxV_re*v0 + VxV_im*u0)
  VecNorm(x, NORM_2, &norm);
  frob += pow(norm, 2.0);      // frob += ||x||^2

  /* obj = 1/2 * || finalstate - gate*rho(0) ||^2 */
  frob *= 1./2.;
  
  // scale by purity of rho(0)
  double purity_rho0 = 0.0;
  double dot = 0.0;
  VecNorm(u0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  VecNorm(v0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  frob = frob / purity_rho0;


  /* Restore vectors from index set */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);

}

void Gate::compare_frobenius_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double frob_bar){

  /* Exit, if this is a dummy gate */
  if (dim_rho == 0) {
    return;
  }

  /* Get real and imag part of final state, initial state, and adjoint */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* Derivative of 1/2 * J */
  double dfb = 1./2. * frob_bar;
  // Derivative of purity scaling 
  double purity_rho0 = 0.0;
  double dot = 0.0;
  VecNorm(u0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  VecNorm(v0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  dfb = dfb / purity_rho0;

  /* Derivative of real part of frobenius norm: 2 * (u - VxV_re*u0 + VxV_im*v0) * dfb */
  MatMult(VxV_re, u0, x);            // x = VxV_re*u0
  VecAYPX(x, -1.0, ufinal);          // x = ufinal - VxV_re*u0 
  MatMultAdd(VxV_im, v0, x, x);      // x = ufinal - VxV_re*u0 + VxV_im*v0

  // add real part in rho0bar, scaled by 2*dfb
  VecISAXPY(rho0_bar, isu, 2.0*dfb, x); 

  /* Derivative of imaginary part of frobenius norm 2 * (v - VxV_re*v0 - VxV_im*u0) * dfb */
  MatMult(VxV_re, v0, x);         // x = VxV_re*v0
  MatMultAdd(VxV_im, u0, x, x);   // x = VxV_re*v0 + VxV_im*u0
  VecAYPX(x, -1.0, vfinal);     // x = vfinal - (VxV_re*v0 + VxV_im*u0)

  // add imaginary part in rho0bar, scaled by 2*dfb
  VecISAXPY(rho0_bar, isv, 2.0*dfb, x); 

  /* Restore final, initial and adjoint state */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);

}

void Gate::compare_trace(const Vec finalstate, const Vec rho0, double& obj){
  obj = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim_rho== 0) {
    return;
  }

  /* Get real and imag part of final state and initial state */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* trace overlap: (VxV_re*u0 - VxV_im*v0)^T u + (VxV_re*v0 + VxV_im*u0)^Tv
              [ + i (VxV_re*u0 - VxV_im*v0)^T v - (VxV_re*v0 + VxV_im*u0)^Tu ]   <- this should be zero!
  */
  double dot;
  double trace = 0.0;

  // first term: (VxV_re*u0 - VxV_im*v0)^T u
  MatMult(VxV_im, v0, x);      
  VecScale(x, -1.0);                // x = - VxV_im*v0
  MatMultAdd(VxV_re, u0, x, x);     // x = VxV_re*u0 - VxV_im*v0
  VecTDot(x, ufinal, &dot);         // dot = (VxV_re*u0 - VxV_im*v0)^T u    
  trace += dot;
  
  // second term: (VxV_re*v0 + VxV_im*u0)^Tv
  MatMult(VxV_im, u0, x);         // x = VxV_im*u0
  MatMultAdd(VxV_re, v0, x, x); // x = VxV_re*v0 + VxV_im*u0
  VecTDot(x, vfinal, &dot);      // dot = (VxV_re*v0 + VxV_im*u0)^T v    
  trace += dot;

  // compute purity of rho(0): Tr(rho(0)^2)
  double purity_rho0 = 0.0;
  VecNorm(u0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  VecNorm(v0, NORM_2, &dot);
  purity_rho0 += dot*dot;

  /* Objective J = 1.0 - Trace(...) */
  obj = 1.0 - trace / purity_rho0;
 
  // // Test: compute constant term  1/2*Tr((Vrho0V^dag)^2)
  // double purity_VrhoV = 0.0;
  // MatMult(VxV_im, v0, x);      
  // VecScale(x, -1.0);           // x = - VxV_im*v0
  // MatMultAdd(VxV_re, u0, x, x);   // x = VxV_re*u0 - VxV_im*v0
  // VecNorm(x, NORM_2, &dot);
  // purity_VrhoV += dot*dot;
  // MatMult(VxV_im, u0, x);         // x = VxV_im*u0
  // MatMultAdd(VxV_re, v0, x, x);   // x = VxV_re*v0 + VxV_im*u0
  // VecNorm(x, NORM_2, &dot);
  // purity_VrhoV += dot*dot;
  // double J_dist = purity_rhoT/2. - trace + purity_VrhoV/2.;
  // printf("J_dist = 1/2 * %f - %f + 1/2 * %f = %1.14e\n", purity_rhoT, trace, purity_VrhoV, J_dist);

  // obj = obj + purity_rhoT / 2. - 0.5;

  /* Restore vectors from index set */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);

  // /* Verify trace overlap */
  // double Jdist = 0.0;
  // compare_frobenius(finalstate, rho0, Jdist);
  // test = test + 1. + Jdist;

  // printf("\n");
  // printf(" J_T:   %1.14e\n", obj);
  // printf(" test:  %1.14e\n", test);
}


void Gate::compare_trace_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double obj_bar){

  /* Exit, if this is a dummy gate */
  if (dim_rho== 0) {
    return;
  }
  double dot;

  /* Get real and imag part of final state and initial state */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* Derivative of 1-trace/purity */
  double dfb = -1.0 * obj_bar;
  // compute purity of rho(0): Tr(rho(0)^2)
  double purity_rho0 = 0.0;
  VecNorm(u0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  VecNorm(v0, NORM_2, &dot);
  purity_rho0 += dot*dot;
  dfb = dfb / purity_rho0;

  // Derivative of first term: -(VxV_re*u0 - VxV_im*v0)*obj_bar
  MatMult(VxV_im, v0, x);      
  VecScale(x, -1.0);              // x = - VxV_im*v0
  MatMultAdd(VxV_re, u0, x, x);  // x = VxV_re*u0 - VxV_im*v0

  /* add real part in rho0bar, scaled by dfb */
  VecISAXPY(rho0_bar, isu, dfb, x); 
  
  // Derivative of second term: -(VxV_re*v0 + VxV_im*u0)*obj_bar
  MatMult(VxV_im, u0, x);         // x = VxV_im*u0
  MatMultAdd(VxV_re, v0, x, x); // x = VxV_re*v0 + VxV_im*u0
  // VecScale(x, dfb);               // x = -(VxV_re*v0 + VxV_im*u0)*obj_bar

  /* add imaginary part in rho0bar, scaled by dfb */
  VecISAXPY(rho0_bar, isv, dfb, x);  

  /* Restore final, initial and adjoint state */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);
}


  XGate::XGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq) : Gate(nlevels, nessential, time, gate_rot_freq) {

  assert(dim_ess == 2);

  /* Fill V_re = Re(V) and V_im = Im(V), V = V_re + iVb */
  /* V_re = 0 1    V_im = 0 0
   *      1 0         0 0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(V_re, 0, 1, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 1, 0, 1.0, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  }

  /* Assemble vectorized rotated target gate \bar VP \kron VP from  V = V_re + i V_im */
  assembleGate();
}

XGate::~XGate() {}

YGate::YGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq ) : Gate(nlevels, nessential, time, gate_rot_freq) {

  assert(dim_ess == 2);
  
  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A = 0 0    B = 0 -1
   *     0 0        1  0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(V_im, 0, 1, -1.0, INSERT_VALUES);
    MatSetValue(V_im, 1, 0,  1.0, INSERT_VALUES);
    MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);
  }

  /* Assemble vectorized rotated arget gate \bar VP \kron VP from  V = V_re + i V_im*/
  assembleGate();
}
YGate::~YGate() {}

ZGate::ZGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq ) : Gate(nlevels, nessential, time, gate_rot_freq) {

  assert(dim_ess == 2);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0 0
   *      0 -1         0 0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(V_im, 0, 0,  1.0, INSERT_VALUES);
    MatSetValue(V_im, 1, 1, -1.0, INSERT_VALUES);
    MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);
  }

  /* Assemble vectorized rotated target gate \bar VP \kron VP from  V = V_re + i V_im*/
  assembleGate();
}

ZGate::~ZGate() {}

HadamardGate::HadamardGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq ) : Gate(nlevels, nessential, time, gate_rot_freq) {

  assert(dim_ess == 2);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0 0
   *      0 -1         0 0
   */
  if (mpirank_petsc == 0) {
    double val = 1./sqrt(2);
    MatSetValue(V_re, 0, 0,  val, INSERT_VALUES);
    MatSetValue(V_re, 0, 1,  val, INSERT_VALUES);
    MatSetValue(V_re, 1, 0,  val, INSERT_VALUES);
    MatSetValue(V_re, 1, 1, -val, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  }

  /* Assemble vectorized rotated target gate \bar VP \kron VP from  V = V_re + i V_im*/
  assembleGate();
}
HadamardGate::~HadamardGate() {}



CNOT::CNOT(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq) : Gate(nlevels, nessential,time, gate_rot_freq) {

  assert(dim_ess == 4);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1 0 0 0   B = 0 0 0 0
   *      0 1 0 0       0 0 0 0
   *      0 0 0 1       0 0 0 0
   *      0 0 1 0       0 0 0 0
   */  if (mpirank_petsc == 0) {
    MatSetValue(V_re, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 1, 1, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 2, 3, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 3, 2, 1.0, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  }

  /* assemble vectorized rotated target gate \bar VP \kron VP from V=V_re + i V_im */
  assembleGate();
}

CNOT::~CNOT(){}


SWAP::SWAP(std::vector<int> nlevels_, std::vector<int> nessential_, double time_, std::vector<double> gate_rot_freq_) : Gate(nlevels_, nessential_, time_, gate_rot_freq_) {
  assert(dim_ess == 4);

  /* Fill lab-frame swap gate in essential dimension system V_re = Re(V), V_im = Im(V) = 0 */
  MatSetValue(V_re, 0, 0, 1.0, INSERT_VALUES);
  MatSetValue(V_re, 1, 2, 1.0, INSERT_VALUES);
  MatSetValue(V_re, 2, 1, 1.0, INSERT_VALUES);
  MatSetValue(V_re, 3, 3, 1.0, INSERT_VALUES);

  MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);

  /* assemble vectorized rotated target gate \bar VP \kron VP from V=V_re + i V_im */
  assembleGate();

}

SWAP::~SWAP(){}

SWAP_0Q::SWAP_0Q(std::vector<int> nlevels_, std::vector<int> nessential_, double time_, std::vector<double> gate_rot_freq_) : Gate(nlevels_, nessential_, time_, gate_rot_freq_) {
  int Q = nlevels.size();  // Number of total oscillators 

  /* Fill lab-frame swap 0<->Q-1 gate in essential dimension system V_re = Re(V), V_im = Im(V) = 0 */

  // diagonal elements. don't swap on states |0xx0> and |1xx1>
  for (int i=0; i< (int) pow(2, Q-2); i++) {
    MatSetValue(V_re, 2*i, 2*i, 1.0, INSERT_VALUES);
  }
  for (int i=(int) pow(2, Q-2); i< pow(2, Q-1); i++) {
    MatSetValue(V_re, 2*i+1, 2*i+1, 1.0, INSERT_VALUES);
  }
  // off-diagonal elements, swap on |0xx1> and |1xx0>
  for (int i=0; i< pow(2, Q-2); i++) {
    MatSetValue(V_re, 2*i + 1, 2*i + (int) pow(2,Q-1), 1.0, INSERT_VALUES);
    MatSetValue(V_re, 2*i + (int) pow(2,Q-1), 2*i + 1, 1.0, INSERT_VALUES);
  }
 

  MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);


  bool isunitary = isUnitary(V_re, V_im);
  if (!isunitary) {
    printf("ERROR: Gate is not unitary!\n");
    exit(1);
  }


  /* assemble vectorized rotated target gate \bar VP \kron VP from V=V_re + i V_im */
  assembleGate();

}

SWAP_0Q::~SWAP_0Q(){}


CQNOT::CQNOT(std::vector<int> nlevels_, std::vector<int> nessential_, double time_, std::vector<double> gate_rot_freq_) : Gate(nlevels_, nessential_, time_, gate_rot_freq_) {

  /* Fill lab-frame CQNOT gate in essential dimension system V_re = Re(V), V_im = Im(V) = 0 */
  /* V = [1 0 0 ...
          0 1 0 ...
          0 0 1 ...
          ..........
                 0 1
                 1 0 ]
  */
  for (int k=0; k<dim_ess-2; k++) { 
    MatSetValue(V_re, k, k, 1.0, INSERT_VALUES);
  }
  MatSetValue(V_re, dim_ess-2, dim_ess-1, 1.0, INSERT_VALUES);
  MatSetValue(V_re, dim_ess-1, dim_ess-2, 1.0, INSERT_VALUES);

  MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);

  /* assemble vectorized rotated target gate \bar VP \kron VP from V=V_re + i V_im */
  assembleGate();
}


CQNOT::~CQNOT(){}