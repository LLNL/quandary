#include "hamiltonian.hpp"


Hamiltonian::Hamiltonian(){
  dim = 0;
  oscil_vec = NULL;
  Re = NULL;
  Im = NULL;
  RHS = NULL;
  dRHSdp = NULL;

}


Hamiltonian::Hamiltonian(int noscillators_, Oscillator** oscil_vec_){
  int ierr;

  noscillators = noscillators_;
  oscil_vec = oscil_vec_;

  /* Dimension of vectorized system: n_1*...*n_q */
  dim = 1;
  for (int iosc = 0; iosc < noscillators_; iosc++) {
    dim *= oscil_vec[iosc]->getNLevels();
  }
  dim = dim*dim; // density matrix: dim \time dim -> vectorized: dim^2

  /* Allocate Re */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,0,NULL,&Re);
  MatSetFromOptions(Re);
  MatSetUp(Re);
  MatSetOption(Re, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  for (int irow = 0; irow < dim; irow++)
  {
    MatSetValue(Re, irow, irow, 0.0, INSERT_VALUES);
  }
  MatAssemblyBegin(Re,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Re,MAT_FINAL_ASSEMBLY);


  /* Allocate Im */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,0,NULL,&Im);
  MatSetFromOptions(Im);
  MatSetUp(Im);
  MatAssemblyBegin(Im,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Im,MAT_FINAL_ASSEMBLY);

  /* Allocate RHS, dimension: 2*dim x 2*dim for the real-valued system */
  MatCreate(PETSC_COMM_WORLD,&RHS);
  MatSetSizes(RHS, PETSC_DECIDE, PETSC_DECIDE,2*dim,2*dim);
  MatSetOptionsPrefix(RHS, "system");
  MatSetFromOptions(RHS);
  MatSetUp(RHS);
  MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY);

  /* Allocate dRHSdp, dimension: 2*dim x 2*nparam*noscil */
  int nparam = oscil_vec[0]->getNParam();
  MatCreate(PETSC_COMM_WORLD,&dRHSdp);
  MatSetSizes(dRHSdp, PETSC_DECIDE, PETSC_DECIDE,2*dim,2*nparam*noscillators);
  MatSetFromOptions(dRHSdp);
  MatSetUp(dRHSdp);
  MatAssemblyBegin(dRHSdp,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(dRHSdp,MAT_FINAL_ASSEMBLY);
}


Hamiltonian::~Hamiltonian(){
  if (dim > 0){
    MatDestroy(&Re);
    MatDestroy(&Im);
    MatDestroy(&RHS);
    MatDestroy(&dRHSdp);
  }
}


int Hamiltonian::getDim(){ return dim; }

bool Hamiltonian::ExactSolution(double t, Vec x) { return false; }

int Hamiltonian::assemble_RHS(double t){
  int ierr;
  int ncol;
  const PetscInt *col_idx;
  const PetscScalar *vals;
  PetscScalar *negvals;
  PetscInt *col_idx_shift;

  /* Allocate tmp vectors */
  ierr = PetscMalloc1(dim, &col_idx_shift);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &negvals);CHKERRQ(ierr);

  /* Set up Jacobian M (=RHS)
   * M(0, 0) =  Re    M(0,1) = -Im
   * M(1, 0) =  Im    M(1,1) = Re
   */
  for (int irow = 0; irow < dim; irow++) {
    PetscInt irow_shift = irow + dim;

    /* Get row in Re */
    ierr = MatGetRow(Re, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + dim;
    }
    // Set A in M: M(0,0) = Re  M(1,1) = Re
    ierr = MatSetValues(RHS,1,&irow,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(RHS,1,&irow_shift,ncol,col_idx_shift,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(Re,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);

    /* Get row in Im */
    ierr = MatGetRow(Im, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + dim;
      negvals[icol] = -vals[icol];
    }
    // Set Im in M: M(1,0) = Im, M(0,1) = -Im
    ierr = MatSetValues(RHS,1,&irow,ncol,col_idx_shift,negvals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(RHS,1,&irow_shift,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(Im,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);
  }

  /* Assemble M */
  ierr = MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // MatView(RHS, PETSC_VIEWER_STDOUT_SELF);

#ifdef SANITY_CHECK
  /* Sanity check. Be careful: This is expensive. */
  // printf("Performing check AntiSymmetric...\n");
  PetscBool isAntiSymmetric;
  MatIsAntiSymmetric(RHS, 0.0, &isAntiSymmetric);
  if (!isAntiSymmetric) printf("%f WARNING: RHS is not symmetric!\n",t);
#endif

  /* Cleanup */
  ierr = PetscFree(col_idx_shift);
  ierr = PetscFree(negvals);

  return 0;
}


int Hamiltonian::assemble_dRHSdp(double t, Vec x) {
  
  return 0;
}

Mat Hamiltonian::getRHS() { return RHS; }
Mat Hamiltonian::getdRHSdp() { return dRHSdp; }


int Hamiltonian::evalObjective(double t, Vec x, double *objective_ptr) {
  
  const PetscScalar *x_ptr;
  VecGetArrayRead(x, &x_ptr);

  *objective_ptr = 200.0 * x_ptr[1]; // TODO: Evaluate objective 

  return 0;
}


int Hamiltonian::evalObjective_diff(double t, Vec x, Vec *lambda, Vec *mu) {
  PetscScalar *x_ptr;

  /* lambda: Derivative of objective wrt x */
  VecZeroEntries(*lambda);  
  VecGetArray(*lambda, &x_ptr);
  x_ptr[1] = 200.0;
  VecRestoreArray(*lambda, &x_ptr);

  /* mu: Derivative of objective wrt parameters */
  VecZeroEntries(*mu);

  return 0;
}

int Hamiltonian::createLoweringOP(int iosc, Mat* loweringOP) {

  /* Get dimensions */
  int nlvls = oscil_vec[iosc]->getNLevels();
  int dim_prekron = 1;
  int dim_postkron = 1;
  for (int j=0; j<noscillators; j++) {
    if (j < iosc) dim_prekron  *= oscil_vec[j]->getNLevels();
    if (j > iosc) dim_postkron *= oscil_vec[j]->getNLevels();
  }
  int dim_lowering = dim_prekron*nlvls*dim_postkron;

  /* create and set lowering operator */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim_lowering,dim_lowering,dim_lowering-1,NULL, loweringOP); 
  for (int i=0; i<dim_prekron; i++) {
    for (int j=0; j<nlvls-1; j++) {
      double val = sqrt(j+1);
      for (int k=0; k<dim_postkron; k++) {
        int row = i * nlvls*dim_postkron + j * dim_postkron + k;
        int col = row + dim_postkron;
        MatSetValue(*loweringOP, row, col, val, INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(*loweringOP, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*loweringOP, MAT_FINAL_ASSEMBLY);

  return dim_lowering;
}



int Hamiltonian::createNumberOP(int iosc, Mat* numberOP) {

  /* Get dimensions */
  int nlvls = oscil_vec[iosc]->getNLevels();
  int dim_prekron = 1;
  int dim_postkron = 1;
  for (int j=0; j<noscillators; j++) {
    if (j < iosc) dim_prekron  *= oscil_vec[j]->getNLevels();
    if (j > iosc) dim_postkron *= oscil_vec[j]->getNLevels();
  }
  int dim_number = dim_prekron*nlvls*dim_postkron;

  /* Create and set number operator */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim_number, dim_number,dim_number,NULL, numberOP); 
  for (int i=0; i<dim_prekron; i++) {
    for (int j=0; j<nlvls; j++) {
      double val = j;
      for (int k=0; k<dim_postkron; k++) {
        int row = i * nlvls*dim_postkron + j * dim_postkron + k;
        int col = row;
        MatSetValue(*numberOP, row, col, val, INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(*numberOP, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*numberOP, MAT_FINAL_ASSEMBLY);
 
  return dim_number;
}


LiouvilleVN::LiouvilleVN() {
  Ad     = NULL;
  Bd     = NULL;
  Ac_vec = NULL;
  Bc_vec = NULL;
  xi     = NULL;
  dRedp = NULL;
  dImdp = NULL;
  rowid = NULL;
  rowid_shift = NULL;

}


LiouvilleVN::LiouvilleVN(double* xi_, int noscillators_, Oscillator** oscil_vec_) 
                :  Hamiltonian(noscillators_, oscil_vec_){
  Mat loweringOP, loweringOP_T;
  Mat numberOP;

  // printf("dim %d\n", dim);

  xi = xi_;

  Ac_vec = new Mat[noscillators_];
  Bc_vec = new Mat[noscillators_];

  /* Allocate constant real and imaginary Hamiltonian Ad = Re(Hd), Bd = Im(Hd) */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,0,NULL,&Ad); // Ad is empty for Liouville VN
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,dim,NULL,&Bd); // Bd is diagonal


  /* Compute building blocks for time-varying Hamiltonian part */
  for (int iosc = 0; iosc < noscillators_; iosc++) {

    /* Get lowering operator */
    int dim_lowering = createLoweringOP(iosc, &loweringOP);
    MatTranspose(loweringOP, MAT_INITIAL_MATRIX, &loweringOP_T);

    /* Compute Ac = I_N \kron (a - a^T) - (a - a^T) \kron I_N */
    MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,4*dim,NULL,&Ac_vec[iosc]); 
    Ikron(loweringOP,   dim_lowering,  1.0, &Ac_vec[iosc], ADD_VALUES);
    Ikron(loweringOP_T, dim_lowering, -1.0, &Ac_vec[iosc], ADD_VALUES);
    kronI(loweringOP_T, dim_lowering, -1.0, &Ac_vec[iosc], ADD_VALUES);
    kronI(loweringOP,   dim_lowering,  1.0, &Ac_vec[iosc], ADD_VALUES);
    MatAssemblyBegin(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    
    /* Compute Bc = - I_N \kron (a + a^T) + (a + a^T) \kron I_N */
    MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,4*dim,NULL,&Bc_vec[iosc]); 
    Ikron(loweringOP,   dim_lowering, -1.0, &Bc_vec[iosc], ADD_VALUES);
    Ikron(loweringOP_T, dim_lowering, -1.0, &Bc_vec[iosc], ADD_VALUES);
    kronI(loweringOP_T, dim_lowering,  1.0, &Bc_vec[iosc], ADD_VALUES);
    kronI(loweringOP,   dim_lowering,  1.0, &Bc_vec[iosc], ADD_VALUES);
    MatAssemblyBegin(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);

    MatDestroy(&loweringOP);
    MatDestroy(&loweringOP_T);
    
  }

  
  /* Compute drift Hamiltonian (Bd only, Ad=0 for Liouville) */
  for (int iosc = 0; iosc < noscillators_; iosc++) {

    Mat tmp, tmp_T;
    Mat numberOPj;
    int dim_number = createNumberOP(iosc, &numberOP);

    /* Diagonal term -xi/2(N_i^2 - N_i) */
    MatMatMult(numberOP, numberOP, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    MatAXPY(tmp, -1.0, numberOP, SAME_NONZERO_PATTERN);
    MatScale(tmp, -xi[iosc]/2.0);

    MatTranspose(tmp, MAT_INITIAL_MATRIX, &tmp_T);
    Ikron(tmp,   dim_number, -1.0, &Bd, ADD_VALUES);
    kronI(tmp_T, dim_number,  1.0, &Bd, ADD_VALUES);

    MatDestroy(&tmp);
    MatDestroy(&tmp_T);


    /* Mixed term -xi(N_i*N_j) for j > i */
    for (int josc = iosc+1; josc < noscillators_; josc++) {

      createNumberOP(iosc, &numberOPj);
      MatMatMult(numberOP, numberOPj, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
      int xi_id = iosc*noscillators_ - iosc*(iosc+1)/2 + josc - (iosc+1);
      MatScale(tmp, -xi[xi_id]);

      MatTranspose(tmp, MAT_INITIAL_MATRIX, &tmp_T);
      Ikron(tmp,   dim_number, -1.0, &Bd, ADD_VALUES);
      kronI(tmp_T, dim_number,  1.0, &Bd, ADD_VALUES);

      MatDestroy(&tmp);
      MatDestroy(&tmp_T);

      MatDestroy(&numberOPj);
    }

    MatDestroy(&numberOP);
  }

  MatAssemblyBegin(Bd, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Bd, MAT_FINAL_ASSEMBLY);

  // MatView(Bd, PETSC_VIEWER_STDOUT_WORLD);

  /* Allocate some auxiliary vectors */
  dRedp = new double[oscil_vec[0]->getNParam()];
  dImdp = new double[oscil_vec[0]->getNParam()];
  rowid = new int[dim];
  rowid_shift = new int[dim];
  for (int i=0; i<dim;i++) {
    rowid[i] = i;
    rowid_shift[i] = i + dim;
  }


}

LiouvilleVN::~LiouvilleVN(){
  MatDestroy(&Ad);
  MatDestroy(&Bd);
  for (int iosc = 0; iosc < noscillators; iosc++) {
    MatDestroy(&Ac_vec[iosc]);
    MatDestroy(&Bc_vec[iosc]);
  }
  delete [] Ac_vec;
  delete [] Bc_vec;
  delete [] dRedp;
  delete [] dImdp;
  delete [] rowid;
  delete [] rowid_shift;

}


int LiouvilleVN::initialCondition(Vec x){
  VecZeroEntries(x); 

  /* Set to first identity vector */
  int idx = 0;
  double val = 1.0;
  VecSetValues(x,1, &idx, &val, INSERT_VALUES);
  
  return 0;
}

int LiouvilleVN::assemble_RHS(double t){
  int ierr;
  double control_Re, control_Im;

  /* Reset */
  ierr = MatZeroEntries(Re);CHKERRQ(ierr);
  ierr = MatZeroEntries(Im);CHKERRQ(ierr);

  /* Time-dependent control part */
  for (int iosc = 0; iosc < noscillators; iosc++) {
    oscil_vec[iosc]->evalControl(t, &control_Re, &control_Im);
    ierr = MatAXPY(Re,control_Im,Ac_vec[iosc],DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = MatAXPY(Im,control_Re,Bc_vec[iosc],DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  }

  /* Constant drift */
  ierr = MatAXPY(Im, 1.0, Bd, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

  /* Set RHS from Re and Im */
  Hamiltonian::assemble_RHS(t);

  return 0;
}

int LiouvilleVN::assemble_dRHSdp(double t, Vec x) {

  Vec u, v;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dim, 0, 1, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dim, dim, 1, &isv);

  /* Get u and v from x = [u v] */
  VecGetSubVector(x, isu, &u);
  VecGetSubVector(x, isv, &v);

  Vec M;
  Vec Acu;
  Vec Acv;
  Vec Bcu;
  Vec Bcv;
  VecDuplicate(u, &M);
  VecDuplicate(u, &Acu);
  VecDuplicate(u, &Acv);
  VecDuplicate(u, &Bcu);
  VecDuplicate(u, &Bcv);
  
  const double *col_ptr;
  int colid;
  int nparam = oscil_vec[0]->getNParam();

  Vec tmp;
  VecDuplicate(u, &tmp);

  /* Reset dRHSdp) */
  MatZeroEntries(dRHSdp);

  /* Loop over oscillators */
  for (int iosc= 0; iosc < noscillators; iosc++){

    /* Evaluate the derivative of the control functions wrt control parameters */
    for (int i=0; i<nparam; i++){
      dRedp[i] = 0.0;
      dImdp[i] = 0.0;
    }
    oscil_vec[iosc]->evalDerivative(t, dRedp, dImdp);

    MatMult(Ac_vec[iosc], u, Acu);
    MatMult(Ac_vec[iosc], v, Acv);
    MatMult(Bc_vec[iosc], u, Bcu);
    MatMult(Bc_vec[iosc], v, Bcv);

    /* Loop over control parameters */
    for (int iparam=0; iparam < nparam; iparam++) {

      /* Derivative wrt paramRe */
      colid = iosc*2*nparam +  iparam;
      VecAXPBY(tmp, -dRedp[iparam], 0.0, Bcv);
      VecGetArrayRead(tmp, &col_ptr);
      MatSetValues(dRHSdp, dim, rowid, 1, &colid, col_ptr, INSERT_VALUES);
      VecRestoreArrayRead(tmp, &col_ptr);

      VecAXPBY(tmp, dRedp[iparam], 0.0, Bcu);
      VecGetArrayRead(tmp, &col_ptr);
      MatSetValues(dRHSdp, dim, rowid_shift, 1, &colid, col_ptr, INSERT_VALUES);
      VecRestoreArrayRead(tmp, &col_ptr);

      /* wrt paramIm */
      colid = iosc*2*nparam + nparam + iparam;
      VecAXPBY(tmp, dImdp[iparam], 0.0, Acu);
      VecGetArrayRead(tmp, &col_ptr);
      MatSetValues(dRHSdp, dim, rowid, 1, &colid, col_ptr, INSERT_VALUES);
      VecRestoreArrayRead(tmp, &col_ptr);

      VecAXPBY(tmp, dImdp[iparam], 0.0, Acv);
      VecGetArrayRead(tmp, &col_ptr);
      MatSetValues(dRHSdp, dim, rowid_shift, 1, &colid, col_ptr, INSERT_VALUES);
      VecRestoreArrayRead(tmp, &col_ptr);
    }
  }

  MatAssemblyBegin(dRHSdp, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(dRHSdp, MAT_FINAL_ASSEMBLY);
  // MatView(dRHSdp, PETSC_VIEWER_STDOUT_WORLD);

  VecDestroy(&M);
  VecDestroy(&Acu);
  VecDestroy(&Acv);
  VecDestroy(&Bcu);
  VecDestroy(&Bcv);
  VecDestroy(&tmp);

  /* Restore y */
  VecRestoreSubVector(x, isu, &u);
  VecRestoreSubVector(x, isv, &v);

  ISDestroy(&isu);
  ISDestroy(&isv);


  return 0;
}


AnalyticHam::AnalyticHam(double* xi_, Oscillator** oscil_vec_) : LiouvilleVN(xi_, 2, oscil_vec_) {}

AnalyticHam::~AnalyticHam() {}


PetscScalar F1_analytic(PetscReal t, PetscReal freq)
{
  PetscScalar f = (1./4.) * (1. - PetscCosScalar(freq * t));
  return f;
}

PetscScalar G2_analytic(PetscReal t,PetscReal freq)
{
  PetscScalar g = (1./4.) * (1. - PetscSinScalar(freq * t));
  return g;
}


PetscScalar dF1_analytic(PetscReal t,PetscReal freq, PetscReal Fbar) {
  PetscScalar dFdp = 1./4. * PetscSinScalar(freq * t) * t * Fbar;
  return dFdp;
}

PetscScalar dG2_analytic(PetscReal t,PetscReal freq, PetscReal Gbar) {
  PetscScalar dGdp = - 1./4. * PetscCosScalar(freq * t) * t * Gbar;
  return dGdp;
}


bool AnalyticHam::ExactSolution(PetscReal t,Vec s)
{
  double f1, f2, g1, g2;
  oscil_vec[0]->getParams(&f1, &g1);
  oscil_vec[1]->getParams(&f2, &g2);

  if (fabs(f1) < 1e-12 || fabs(g2) < 1e-12){
    printf("\n ERROR: Can't use 0.0 for FunctionOscillatorFrequency!\n");
    exit(1);
  }

  PetscScalar    *s_localptr;

  /* Get a pointer to vector data. */
  VecGetArray(s,&s_localptr);;

  /* Write the solution into the array locations.
   *  Alternatively, we could use VecSetValues() or VecSetValuesLocal(). */
  PetscScalar phi = (1./4.) * (t - (1./ f1)*PetscSinScalar(f1*t));
  PetscScalar theta = (1./4.) * (t + (1./ g2)*PetscCosScalar(g2*t) - 1.);
  PetscScalar cosphi = PetscCosScalar(phi);
  PetscScalar costheta = PetscCosScalar(theta);
  PetscScalar sinphi = PetscSinScalar(phi);
  PetscScalar sintheta = PetscSinScalar(theta);
  

  // /* Real part */
  s_localptr[0] = cosphi*costheta*cosphi*costheta;
  s_localptr[1] = -1.*cosphi*sintheta*cosphi*costheta;
  s_localptr[2] = 0.;
  s_localptr[3] = 0.;
  s_localptr[4] = -1.*cosphi*costheta*cosphi*sintheta;
  s_localptr[5] = cosphi*sintheta*cosphi*sintheta;
  s_localptr[6] = 0.;
  s_localptr[7] = 0.;
  s_localptr[8] = 0.;
  s_localptr[9] = 0.;
  s_localptr[10] = sinphi*costheta*sinphi*costheta;
  s_localptr[11] = -1.*sinphi*sintheta*sinphi*costheta;
  s_localptr[12] = 0.;
  s_localptr[13] = 0.;
  s_localptr[14] = -1.*sinphi*costheta*sinphi*sintheta;
  s_localptr[15] = sinphi*sintheta*sinphi*sintheta;
  /* Imaginary part */
  s_localptr[16] = 0.;
  s_localptr[17] = 0.;
  s_localptr[18] = - sinphi*costheta*cosphi*costheta;
  s_localptr[19] = sinphi*sintheta*cosphi*costheta;
  s_localptr[20] = 0.;
  s_localptr[21] = 0.;
  s_localptr[22] = sinphi*costheta*cosphi*sintheta;
  s_localptr[23] = - sinphi*sintheta*cosphi*sintheta;
  s_localptr[24] = cosphi*costheta*sinphi*costheta;
  s_localptr[25] = - cosphi*sintheta*sinphi*costheta;
  s_localptr[26] = 0.;
  s_localptr[27] = 0.;
  s_localptr[28] = - cosphi*costheta*sinphi*sintheta;
  s_localptr[29] = cosphi*sintheta*sinphi*sintheta;
  s_localptr[30] = 0.;
  s_localptr[31] = 0.;

  // /* Restore solution vector */
  VecRestoreArray(s,&s_localptr);

  return true;
}


PetscErrorCode AnalyticHam::initialCondition(Vec x)
{
  // ExactSolution(0,x);
  // VecView(x, PETSC_VIEWER_STDOUT_WORLD);

  VecZeroEntries(x);
  PetscScalar *x_ptr;
  VecGetArray(x, &x_ptr);
  x_ptr[0] = 1.0;
  VecRestoreArray(x, &x_ptr);
  return 0;
}


