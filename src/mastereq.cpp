#include "mastereq.hpp"


MasterEq::MasterEq(){
  dim = 0;
  oscil_vec = NULL;
  Re = NULL;
  Im = NULL;
  RHS = NULL;
  dRHSdp = NULL;
  Ad     = NULL;
  Bd     = NULL;
  Ac_vec = NULL;
  Bc_vec = NULL;
  dRedp = NULL;
  dImdp = NULL;
  rowid = NULL;
  rowid_shift = NULL;


}


MasterEq::MasterEq(int noscillators_, Oscillator** oscil_vec_, const std::vector<double> xi_, LindbladType lindbladtype, const std::vector<double> gamma_) {
  int ierr;

  noscillators = noscillators_;
  oscil_vec = oscil_vec_;
  xi = xi_;
  gamma = gamma_;
  assert(xi.size() == (noscillators_+1) * noscillators_ / 2);


  /* Dimension of vectorized system: n_1*...*n_q */
  dim = 1;
  for (int iosc = 0; iosc < noscillators_; iosc++) {
    dim *= oscil_vec[iosc]->getNLevels();
  }
  dim = dim*dim; // density matrix: N \times N -> vectorized: N^2

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

  /* Allocate auxiliary vectors */
  ierr = PetscMalloc1(dim, &col_idx_shift);
  ierr = PetscMalloc1(dim, &negvals);
  
 /* Allocate time-varying building blocks */
  Mat loweringOP, loweringOP_T;
  Mat numberOP;
  Ac_vec = new Mat[noscillators_];
  Bc_vec = new Mat[noscillators_];


  /* Compute building blocks */
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


  /* Allocate and compute imag drift part Bd = Hd */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,1,NULL,&Bd); 
  int xi_id = 0;
  for (int iosc = 0; iosc < noscillators_; iosc++) {
    Mat tmp, tmp_T;
    Mat numberOPj;
    int dim_number = createNumberOP(iosc, &numberOP);

    /* Diagonal term - 2* PI * xi/2 *(N_i^2 - N_i) */
    MatMatMult(numberOP, numberOP, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    MatAXPY(tmp, -1.0, numberOP, SAME_NONZERO_PATTERN);
    MatScale(tmp, -xi[xi_id] * M_PI);
    xi_id++;

    MatTranspose(tmp, MAT_INITIAL_MATRIX, &tmp_T);
    Ikron(tmp,   dim_number, -1.0, &Bd, ADD_VALUES);
    kronI(tmp_T, dim_number,  1.0, &Bd, ADD_VALUES);

    MatDestroy(&tmp);
    MatDestroy(&tmp_T);

    /* Mixed term -xi * 2 * PI * (N_i*N_j) for j > i */
    for (int josc = iosc+1; josc < noscillators_; josc++) {

      createNumberOP(josc, &numberOPj);
      MatMatMult(numberOP, numberOPj, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
      MatScale(tmp, -xi[xi_id] * 2.0 * M_PI);
      xi_id++;

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

  /* Allocate and compute real drift part Ad = Lindblad */
  Mat L, tmp;
  int dim_L;
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,2,NULL,&Ad); 
  for (int iosc = 0; iosc < noscillators_; iosc++) {

    /* Get lowering operator */
    switch (lindbladtype)  {
      case NONE:
        continue;
        break;
      case DECAY: 
        dim_L = createLoweringOP(iosc, &L);
        break;
      case DEPHASING:
        dim_L = createNumberOP(iosc, &L);
        break;
      default:
        printf("ERROR! Wrong lindblad type: %d\n", lindbladtype);
        exit(1);
    }
    
    /* --- Ad = gamma_j L \kron L - gamma_j I\dron L^TL + L^TL\dron I --- */
    /* First term Ad = gamma_j * L \kron L */
    AkronB(dim_L, L, L, gamma[iosc], &Ad, ADD_VALUES);
    /* Second term Ad += - gamma_j/2 I_n  \kron L^TL and third term Ad += - gamma_j/2  L^TL \kron I_n */
    MatTransposeMatMult(L, L, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    Ikron(tmp, dim_L, -gamma[iosc]/2, &Ad, ADD_VALUES);
    kronI(tmp, dim_L, -gamma[iosc]/2, &Ad, ADD_VALUES);
    MatDestroy(&tmp);
    MatDestroy(&L);
  }
  MatAssemblyBegin(Ad, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ad, MAT_FINAL_ASSEMBLY);

  /* Create vector strides for accessing Re and Im part for x = [u v] */
  ISCreateStride(PETSC_COMM_WORLD, dim, 0, 1, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dim, dim, 1, &isv);
 
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


MasterEq::~MasterEq(){
  if (dim > 0){
    MatDestroy(&Re);
    MatDestroy(&Im);
    MatDestroy(&RHS);
    MatDestroy(&dRHSdp);
    PetscFree(col_idx_shift);
    PetscFree(negvals);

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

    ISDestroy(&isu);
    ISDestroy(&isv);
  }
}


int MasterEq::getDim(){ return dim; }

int MasterEq::getNOscillators() { return noscillators; }

Oscillator* MasterEq::getOscillator(int i) { return oscil_vec[i]; }

bool MasterEq::ExactSolution(double t, Vec x) { return false; }

int MasterEq::assemble_RHS(double t){
  int ierr;
  int ncol;
  const PetscInt *col_idx;
  const PetscScalar *vals;
  double control_Re, control_Im;

  /* Reset */
  ierr = MatZeroEntries(Re);CHKERRQ(ierr);
  ierr = MatZeroEntries(Im);CHKERRQ(ierr);
  ierr = MatZeroEntries(RHS);CHKERRQ(ierr);

  /* Time-dependent control part */
  for (int iosc = 0; iosc < noscillators; iosc++) {
    oscil_vec[iosc]->evalControl(t, &control_Re, &control_Im);
    ierr = MatAXPY(Re,control_Im,Ac_vec[iosc],DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = MatAXPY(Im,control_Re,Bc_vec[iosc],DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  }

  /* Constant drift parts */
  ierr = MatAXPY(Re, 1.0, Ad, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatAXPY(Im, 1.0, Bd, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);


  /* Set up Jacobian M (=RHS) from Re and Im
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
  if (!isAntiSymmetric) printf("%f WARNING: RHS is not antisymmetric!\n",t);
#endif

  return 0;
}



Mat MasterEq::getRHS() { return RHS; }
Mat MasterEq::getdRHSdp() { return dRHSdp; }



int MasterEq::createLoweringOP(int iosc, Mat* loweringOP) {

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



int MasterEq::createNumberOP(int iosc, Mat* numberOP) {

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


int MasterEq::initialCondition(int iinit, Vec x){

  /* Set to i-th unit vector */
  int idx = iinit;
  double val = 1.0;
  VecZeroEntries(x); 
  VecSetValues(x,1, &idx, &val, INSERT_VALUES);
  // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  
  return 0;
}


int MasterEq::assemble_dRHSdp(double t, Vec x) {

  /* Get real and imaginary part from x = [u v] */
  Vec u, v;
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

  /* Restore x */
  VecRestoreSubVector(x, isu, &u);
  VecRestoreSubVector(x, isv, &v);

  return 0;
}

