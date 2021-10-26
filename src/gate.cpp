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
  PetscInt ilow, iupp;
  MatGetOwnershipRange(VxV_re, &ilow, &iupp);
  PetscInt dimis = iupp - ilow;
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
  PetscInt *cols;
  PetscMalloc1(dim_ess, &out_re); 
  PetscMalloc1(dim_ess, &out_im); 
  PetscMalloc1(dim_ess, &cols); 
  // get the frequency of the diagonal scaling e^{iwt} for each row rotation matrix R=R1\otimes R2\otimes...
  for (PetscInt row=0; row<dim_ess; row++){
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
  PetscInt ilow, iupp;
  MatGetOwnershipRange(VxV_re, &ilow, &iupp);
  double val;
  double vre_ij, vim_ij;
  double vre_rc, vim_rc;
  // iterate over rows of V_e (essential dimension gate)
  for (PetscInt row_f=0;row_f<dim_rho; row_f++) {
    if (isEssential(row_f, nlevels, nessential)) { // place \bar v_xx*V_f blocks for all cols in V_e[row_e]
      PetscInt row_e = mapFullToEss(row_f, nlevels, nessential);
      assert(row_f == mapEssToFull(row_e, nlevels, nessential));
      // iterate over columns in this row_e
      for (PetscInt col_e=0; col_e<dim_ess; col_e++) {
        vre_ij = 0.0; vim_ij = 0.0;
        MatGetValues(V_re, 1, &row_e, 1, &col_e, &vre_ij);
        MatGetValues(V_im, 1, &row_e, 1, &col_e, &vim_ij);
        // for all nonzeros in this row, place block \bar Ve_{i,j} * (V_f) at starting position G[a,b]
        if (fabs(vre_ij) > 1e-14 || fabs(vim_ij) > 1e-14 ) {
          int a = row_f * dim_rho;
          int b = mapEssToFull(col_e, nlevels, nessential) * dim_rho;
          // iterate over rows in V_f
          for (PetscInt r=0; r<dim_rho; r++) {
            PetscInt rowout = a + r;  // row in G
            if (ilow <= rowout && rowout < iupp) {
              if (isEssential(r, nlevels, nessential)){ // place ve_ij*ve_rc at G[a+r, b+map(ce)]
                PetscInt re = mapFullToEss(r, nlevels, nessential);
                for (PetscInt ce=0; ce<dim_ess; ce++) {
                  PetscInt colout = b + mapEssToFull(ce, nlevels, nessential); // column in G
                  vre_rc = 0.0; vim_rc = 0.0;
                  MatGetValues(V_re, 1, &re, 1, &ce, &vre_rc);
                  MatGetValues(V_im, 1, &re, 1, &ce, &vim_rc);
                  val = vre_ij*vre_rc + vim_ij*vim_rc;
                  if (fabs(val) > 1e-14) MatSetValue(VxV_re, rowout, colout, val, INSERT_VALUES);
                  val = vre_ij*vim_rc - vim_ij*vre_rc;
                  if (fabs(val) > 1e-14) MatSetValue(VxV_im, rowout, colout, val, INSERT_VALUES);
                }  
              } else { // place ve_ij*1.0 at G[a+row, a+row]
              PetscInt colout = b + r;
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
        PetscInt rowout = a + r;  // row in G
        if (ilow <= rowout && rowout < iupp) {
          if (isEssential(r, nlevels, nessential)){ // place ve_rc at G[a+r, a+map(ce)]
            PetscInt re = mapFullToEss(r, nlevels, nessential);
            for (PetscInt ce=0; ce<dim_ess; ce++) {
              PetscInt colout = a + mapEssToFull(ce, nlevels, nessential); // column in G
              vre_rc = 0.0; vim_rc = 0.0;
              MatGetValues(V_re, 1, &re, 1, &ce, &vre_rc);
              MatGetValues(V_im, 1, &re, 1, &ce, &vim_rc);
              val = vre_rc;
              if (fabs(val) > 1e-14) MatSetValue(VxV_re, rowout, colout, val, INSERT_VALUES);
              val = vim_rc;
              if (fabs(val) > 1e-14) MatSetValue(VxV_im, rowout, colout, val, INSERT_VALUES);
            }  
          } else { // place 1.0 at G[a+r, a+r]
              PetscInt colout = a + r;
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



void Gate::applyGate(const Vec state, Vec VrhoV){
  /* Exit, if this is a dummy gate */
  if (dim_rho == 0) return;

  /* Get real and imag part of the state q = u + iv */
  Vec u, v;
  VecGetSubVector(state, isu, &u);
  VecGetSubVector(state, isv, &v);

  /* (a) Real part Re(VxV q) = VxV_re u - VxV_im v */
  MatMult(VxV_im, v, x);            // 
  VecScale(x, -1.0);                // x = -VxV_im * v 
  MatMultAdd(VxV_re, u, x, x);      // x += VxV_re * u
  VecISCopy(VrhoV, isu, SCATTER_FORWARD, x); 

  /* (b) Imaginary part Im(VxV q) = VxV_re v + VxV_im u */
  MatMult(VxV_re, v, x);            // x  = VxV_re * v
  MatMultAdd(VxV_im, u, x, x);      // x += VxV_im * u
  VecISCopy(VrhoV, isv, SCATTER_FORWARD, x); 

  /* Restore state from index set */
  VecRestoreSubVector(state, isu, &u);
  VecRestoreSubVector(state, isv, &v);
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