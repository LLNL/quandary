#include "mastereq.hpp"


MasterEq::MasterEq(){
  dim = 0;
  oscil_vec = NULL;
  Re = NULL;
  Im = NULL;
  RHS = NULL;
  Ad     = NULL;
  Bd     = NULL;
  Ac_vec = NULL;
  Bc_vec = NULL;
  dRedp = NULL;
  dImdp = NULL;
  rowid = NULL;
  rowid_shift = NULL;


}


MasterEq::MasterEq(int noscillators_, Oscillator** oscil_vec_, const std::vector<double> xi_, LindbladType lindbladtype, InitialConditionType initcond_type_, const std::vector<double> collapse_time_) {
  int ierr;

  noscillators = noscillators_;
  oscil_vec = oscil_vec_;
  xi = xi_;
  collapse_time = collapse_time_;
  initcond_type = initcond_type_;
  assert(xi.size() >= (noscillators_+1) * noscillators_ / 2);
  assert(collapse_time.size() >= 2*noscillators);


  /* Dimension of vectorized system: (n_1*...*n_q^2 */
  dim = 1;
  for (int iosc = 0; iosc < noscillators_; iosc++) {
    dim *= oscil_vec[iosc]->getNLevels();
  }
  int dimmat = dim;
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

    /* Get lowering operator a = I_(n_1) \kron ... \kron a^(n_k) \kron ... \kron I_(n_q) */
    loweringOP = oscil_vec[iosc]->getLoweringOP();
    MatTranspose(loweringOP, MAT_INITIAL_MATRIX, &loweringOP_T);

    /* Compute Ac = I_N \kron (a - a^T) - (a - a^T) \kron I_N */
    MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,4,NULL,&Ac_vec[iosc]); 
    Ikron(loweringOP,   dimmat,  1.0, &Ac_vec[iosc], ADD_VALUES);
    Ikron(loweringOP_T, dimmat, -1.0, &Ac_vec[iosc], ADD_VALUES);
    kronI(loweringOP_T, dimmat, -1.0, &Ac_vec[iosc], ADD_VALUES);
    kronI(loweringOP,   dimmat,  1.0, &Ac_vec[iosc], ADD_VALUES);
    MatAssemblyBegin(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    
    /* Compute Bc = - I_N \kron (a + a^T) + (a + a^T) \kron I_N */
    MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,4,NULL,&Bc_vec[iosc]); 
    Ikron(loweringOP,   dimmat, -1.0, &Bc_vec[iosc], ADD_VALUES);
    Ikron(loweringOP_T, dimmat, -1.0, &Bc_vec[iosc], ADD_VALUES);
    kronI(loweringOP_T, dimmat,  1.0, &Bc_vec[iosc], ADD_VALUES);
    kronI(loweringOP,   dimmat,  1.0, &Bc_vec[iosc], ADD_VALUES);
    MatAssemblyBegin(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);

    MatDestroy(&loweringOP_T);
  }

  /* Allocate and compute imag drift part Bd = Hd */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim,dim,1,NULL,&Bd); 
  int xi_id = 0;
  for (int iosc = 0; iosc < noscillators_; iosc++) {
    Mat tmp, tmp_T;
    Mat numberOPj;
    
    /* Create number operator */
    numberOP = oscil_vec[iosc]->getNumberOP();

    /* Diagonal term - 2* PI * xi/2 *(N_i^2 - N_i) */
    MatMatMult(numberOP, numberOP, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    MatAXPY(tmp, -1.0, numberOP, SAME_NONZERO_PATTERN);
    MatScale(tmp, -xi[xi_id] * M_PI);
    xi_id++;

    MatTranspose(tmp, MAT_INITIAL_MATRIX, &tmp_T);
    Ikron(tmp,   dimmat, -1.0, &Bd, ADD_VALUES);
    kronI(tmp_T, dimmat,  1.0, &Bd, ADD_VALUES);

    MatDestroy(&tmp);
    MatDestroy(&tmp_T);

    /* Mixed term -xi * 2 * PI * (N_i*N_j) for j > i */
    for (int josc = iosc+1; josc < noscillators_; josc++) {
      numberOPj = oscil_vec[josc]->getNumberOP();
      MatMatMult(numberOP, numberOPj, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
      MatScale(tmp, -xi[xi_id] * 2.0 * M_PI);
      xi_id++;

      MatTranspose(tmp, MAT_INITIAL_MATRIX, &tmp_T);
      Ikron(tmp,   dimmat, -1.0, &Bd, ADD_VALUES);
      kronI(tmp_T, dimmat,  1.0, &Bd, ADD_VALUES);

      MatDestroy(&tmp);
      MatDestroy(&tmp_T);
    }
  }
  MatAssemblyBegin(Bd, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Bd, MAT_FINAL_ASSEMBLY);

  /* Allocate and compute real drift part Ad = Lindblad */
  Mat L1, L2, tmp;
  bool addT1, addT2;
  MatCreate(PETSC_COMM_WORLD, &Ad);
  MatSetSizes(Ad, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatSetUp(Ad);
  for (int iosc = 0; iosc < noscillators_; iosc++) {
  
    switch (lindbladtype)  {
      case NONE:
        continue;
        break;
      case DECAY: 
        L1 = oscil_vec[iosc]->getLoweringOP();
        addT1 = true;
        addT2 = false;
        break;
      case DEPHASE:
        L2 = oscil_vec[iosc]->getNumberOP();
        addT1 = false;
        addT2 = true;
        break;
      case BOTH:
        L1 = oscil_vec[iosc]->getLoweringOP();
        L2 = oscil_vec[iosc]->getNumberOP();
        addT1 = true;
        addT2 = true;
        break;
      default:
        printf("ERROR! Wrong lindblad type: %d\n", lindbladtype);
        exit(1);
    }

    /* --- Adding T1-DECAY (L1 = a_j) for oscillator j --- */
    if (addT1) {
      if (collapse_time[iosc*2] < 1e-12) { 
        printf("Check lindblad settings! Requested collapse time is zero.\n"); exit(1);
      }
      double gamma = 1./(collapse_time[iosc*2]);
      /* Ad += gamma_j * L \kron L */
      AkronB(dimmat, L1, L1, gamma, &Ad, ADD_VALUES);
      /* Ad += - gamma_j/2  I_n  \kron L^TL  */
      /* Ad += - gamma_j/2  L^TL \kron I_n */
      MatTransposeMatMult(L1, L1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
      Ikron(tmp, dimmat, -gamma/2, &Ad, ADD_VALUES);
      kronI(tmp, dimmat, -gamma/2, &Ad, ADD_VALUES);
      MatDestroy(&tmp);
    }

    /* --- Adding T2-Dephasing (L1 = a_j^\dag a_j) for oscillator j --- */
    if (addT2) {
      if (collapse_time[iosc*2+1] < 1e-12) { 
        printf("Check lindblad settings! Requested collapse time is zero.\n"); exit(1);
      }
      double gamma = 1./(collapse_time[iosc*2+1]);
      /* Ad += 1./gamma_j * L \kron L */
      AkronB(dimmat, L2, L2, gamma, &Ad, ADD_VALUES);
      /* Ad += - gamma_j/2  I_n  \kron L^TL  */
      /* Ad += - gamma_j/2  L^TL \kron I_n */
      MatTransposeMatMult(L2, L2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
      Ikron(tmp, dimmat, -gamma/2, &Ad, ADD_VALUES);
      kronI(tmp, dimmat, -gamma/2, &Ad, ADD_VALUES);
      MatDestroy(&tmp);
    }
  }
  MatAssemblyBegin(Ad, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ad, MAT_FINAL_ASSEMBLY);

  /* Create vector strides for accessing Re and Im part for x = [u v] */
  ISCreateStride(PETSC_COMM_WORLD, dim, 0, 1, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dim, dim, 1, &isv);
 
  /* Allocate some auxiliary vectors */
  dRedp = new double[oscil_vec[0]->getNParams()];
  dImdp = new double[oscil_vec[0]->getNParams()];
  cols = new int[oscil_vec[0]->getNParams()];
  vals = new double [oscil_vec[0]->getNParams()];
  rowid = new int[dim];
  rowid_shift = new int[dim];
  for (int i=0; i<dim;i++) {
    rowid[i] = i;
    rowid_shift[i] = i + dim;
  }

  MatCreateVecs(Ac_vec[0], &Acu, NULL);
  MatCreateVecs(Ac_vec[0], &Acv, NULL);
  MatCreateVecs(Bc_vec[0], &Bcu, NULL);
  MatCreateVecs(Bc_vec[0], &Bcv, NULL);
  MatCreateVecs(Bc_vec[0], &auxil, NULL);

}


MasterEq::~MasterEq(){
  if (dim > 0){
    MatDestroy(&Re);
    MatDestroy(&Im);
    MatDestroy(&RHS);
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
    delete [] cols;
    delete [] vals;
    delete [] rowid;
    delete [] rowid_shift;

    ISDestroy(&isu);
    ISDestroy(&isv);

    VecDestroy(&Acu);
    VecDestroy(&Acv);
    VecDestroy(&Bcu);
    VecDestroy(&Bcv);
    VecDestroy(&auxil);
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

  return 0;
}



Mat MasterEq::getRHS() { return RHS; }


void MasterEq::reducedDensity(Vec fulldensitymatrix, Vec *reduced, int dim_pre, int dim_post, int dim_reduced) {

  /* sanity test */
  int dimmat = dim_pre * dim_reduced * dim_post;
  assert ( (int) pow(dimmat,2) == dim);

  /* Iterate over reduced density matrix elements */
  for (int i=0; i<dim_reduced; i++) {
    for (int j=0; j<dim_reduced; j++) {
      /* Iterate over all dim_pre blocks of size n_k * dim_post */
      for (int l = 0; l < dim_pre; l++) {
        int blockstartID = l * dim_reduced * dim_post; // Go to beginning of block 
        /* iterate over elements in this block */
        for (int m=0; m<dim_post; m++) {
          int rho_row = blockstartID + i * dim_post + m;
          int rho_col = blockstartID + j * dim_post + m;
          int rho_vecID_re = rho_col * dimmat + rho_row;
          int rho_vecID_im = rho_vecID_re + (int) pow(dimmat,2);
          /* Get real and imaginary part from full density matrix */
          double re, im;
          VecGetValues(fulldensitymatrix, 1, &rho_vecID_re, &re);
          VecGetValues(fulldensitymatrix, 1, &rho_vecID_im, &im);
          /* Set real and imaginary part for reduced density matrix */
          int out_vecID_re = j * dim_reduced + i;
          int out_vecID_im = out_vecID_re + (int) pow(dim_reduced,2);
          VecSetValues( *reduced, 1, &out_vecID_re, &re, ADD_VALUES);
          VecSetValues( *reduced, 1, &out_vecID_im, &im, ADD_VALUES);
        }
      }
    }
  }
  VecAssemblyBegin(*reduced);
  VecAssemblyEnd(*reduced);

}


void MasterEq::reducedDensity_diff(Vec fulldens_bar, Vec reduced_bar, int dim_pre, int dim_post, int dim_reduced) {
  /* sanity test */
  int dimmat = dim_pre * dim_reduced * dim_post;
  assert ( (int) pow(dimmat,2) == dim);

 /* Iterate over reduced density matrix elements */
  for (int i=0; i<dim_reduced; i++) {
    for (int j=0; j<dim_reduced; j++) {
      /* Iterate over all dim_pre blocks of size n_k * dim_post */
      for (int l = 0; l < dim_pre; l++) {
        int blockstartID = l * dim_reduced * dim_post; // Go to beginning of block 
        /* iterate over elements in this block */
        for (int m=0; m<dim_post; m++) {
          /* Get value from reduced redens_bar */
          int vecID_re = j * dim_reduced + i;
          int vecID_im = vecID_re + (int) pow(dim_reduced,2);
          double re, im;
          VecGetValues( reduced_bar, 1, &vecID_re, &re);
          VecGetValues( reduced_bar, 1, &vecID_im, &im);

          /* Set values into full density_bar  */
          int rho_row = blockstartID + i * dim_post + m;
          int rho_col = blockstartID + j * dim_post + m;
          int rho_vecID_re = rho_col * dimmat + rho_row;
          int rho_vecID_im = rho_col * dimmat + rho_row + dim;

          /* Set derivative */
          VecSetValues(fulldens_bar, 1, &rho_vecID_re, &re, ADD_VALUES);
          VecSetValues(fulldens_bar, 1, &rho_vecID_im, &im, ADD_VALUES);
        }
      }
    }
  }
  VecAssemblyBegin(fulldens_bar); VecAssemblyEnd(fulldens_bar);

}


/* grad += RHS(x)^T * xbar  */
void MasterEq::computedRHSdp(double t, Vec x, Vec xbar, double alpha, Vec grad) {

  /* Get real and imaginary part from x = [u v] and x_bar */
  Vec u, v, ubar, vbar;
  VecGetSubVector(x, isu, &u);
  VecGetSubVector(x, isv, &v);
  VecGetSubVector(xbar, isu, &ubar);
  VecGetSubVector(xbar, isv, &vbar);

  /* Get number of control parameters for this oscillator */
  int nparam = oscil_vec[0]->getNParams(); // TODO: THIS WORKS ONLY IF ALL OSCIL HAVE SAME NUMBER!

  /* Loop over oscillators */
  for (int iosc= 0; iosc < noscillators; iosc++){

    /* Evaluate the derivative of the control functions wrt control parameters */
    for (int i=0; i<nparam; i++){
      dRedp[i] = 0.0;
      dImdp[i] = 0.0;
    }
    oscil_vec[iosc]->evalDerivative(t, dRedp, dImdp);

    /* Compute RHS matrix vector products */
    MatMult(Ac_vec[iosc], u, Acu);
    MatMult(Ac_vec[iosc], v, Acv);
    MatMult(Bc_vec[iosc], u, Bcu);
    MatMult(Bc_vec[iosc], v, Bcv);

    /* Compute terms in RHS(x)^T xbar */
    double uAubar, vAvbar, vBubar, uBvbar;
    VecDot(Acu, ubar, &uAubar);
    VecDot(Acv, vbar, &vAvbar);
    VecDot(Bcu, vbar, &uBvbar);
    VecDot(Bcv, ubar, &vBubar);

    /* Set gradient terms for each control parameter */
    for (int iparam=0; iparam < nparam; iparam++) {
      vals[iparam] = (uAubar + vAvbar) * dImdp[iparam] + ( -vBubar + uBvbar) * dRedp[iparam];
      cols[iparam] = iosc * nparam + iparam;
    }
    VecSetValues(grad, nparam, cols, vals, ADD_VALUES);
  }
  VecAssemblyBegin(grad); 
  VecAssemblyEnd(grad);

  /* Restore x */
  VecRestoreSubVector(x, isu, &u);
  VecRestoreSubVector(x, isv, &v);
  VecRestoreSubVector(xbar, isu, &ubar);
  VecRestoreSubVector(xbar, isv, &vbar);

}

void MasterEq::setControlAmplitudes(Vec x) {

  const PetscScalar* ptr;
  VecGetArrayRead(x, &ptr);

  /* Pass design vector x to oscillators */
  int shift=0;
  for (int ioscil = 0; ioscil < getNOscillators(); ioscil++) {
    /* Design storage: x = (params_oscil0, params_oscil2, ... ) */
    getOscillator(ioscil)->setParams(ptr + shift); 
    shift += getOscillator(ioscil)->getNParams();
  }
  VecRestoreArrayRead(x, &ptr);
}


int MasterEq::getRhoT0(int iinit, std::vector<int> oscilIDs, int ninit, Vec rho0){

  int dim_post;
  int initID = -1;    // Output: ID for this initial condition */
  int dim_rho = (int) sqrt(dim); // N


  /* Switch over type of initial condition */
  switch (initcond_type) {

    case FROMFILE:
      /* Do nothing. Init cond is already stored */
      break;

    case PURE:
      /* Do nothing. Init cond is already stored */
      break;

    case DIAGONAL:
      int row, diagelem;

      /* Reset the initial conditions */
      VecZeroEntries(rho0); 

      /* Get dimension of partial system behind last oscillator ID */
      dim_post = 1;
      for (int k = oscilIDs[oscilIDs.size()-1] + 1; k < getNOscillators(); k++) {
        dim_post *= getOscillator(k)->getNLevels();
      }

      /* Compute index of the nonzero element in rho_m(0) = E_pre \otimes |m><m| \otimes E_post */
      diagelem = iinit * dim_post;
      // /* Position in vectorized q(0) */
      row = diagelem * dim_rho + diagelem;

      /* Assemble */
      VecSetValue(rho0, row, 1.0, INSERT_VALUES);
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);

      /* Set initial conditon ID */
      initID = iinit * ( (int) sqrt(ninit) ) + iinit;

      break;

    case BASIS:

      /* Reset the initial conditions */
      VecZeroEntries(rho0); 

      /* Get dimension of partial system behind last oscillator ID */
      dim_post = 1;
      for (int k = oscilIDs[oscilIDs.size()-1] + 1; k < getNOscillators(); k++) {
        dim_post *= getOscillator(k)->getNLevels();
      }

      /* Get index (k,j) of basis element B_{k,j} for this initial condition index iinit */
      int k, j;
      k = iinit % ( (int) sqrt(ninit) );
      j = (int) iinit / ( (int) sqrt(ninit) );   

      if (k == j) {
        /* B_{kk} = E_{kk} -> set only one element at (k,k) */
        int elemID = j * dim_post * dim_rho + k * dim_post;
        double val = 1.0;
        VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
      } else {
      //   /* B_{kj} contains four non-zeros, two per row */
        int* rows = new int[4];
        double* vals = new double[4];

        rows[0] = k * dim_post * dim_rho + k * dim_post; // (k,k)
        rows[1] = j * dim_post * dim_rho + j * dim_post; // (j,j)
        rows[2] = j * dim_post * dim_rho + k * dim_post; // (k,j)
        rows[3] = k * dim_post * dim_rho + j * dim_post; // (j,k)

        if (k < j) { // B_{kj} = 1/2(E_kk + E_jj) + 1/2(E_kj + E_jk)
          vals[0] = 0.5;
          vals[1] = 0.5;
          vals[2] = 0.5;
          vals[3] = 0.5;
          VecSetValues(rho0, 4, rows, vals, INSERT_VALUES);
        } else {  // B_{kj} = 1/2(E_kk + E_jj) + i/2(E_jk - E_kj)
          vals[0] = 0.5;
          vals[1] = 0.5;
          VecSetValues(rho0, 2, rows, vals, INSERT_VALUES); // diagonal, real
          vals[0] = -0.5;
          vals[1] = 0.5;
          rows[2] = rows[2] + dim; // Shift to imaginary 
          rows[3] = rows[3] + dim;
          VecSetValues(rho0, 2, rows+2, vals, INSERT_VALUES); // off-diagonals, imaginary
        }

        VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);
        delete [] rows; 
        delete [] vals;
      }

      /* Set initial condition ID */
      initID = j * ( (int) sqrt(ninit)) + k;

      break;

    default:
      printf("ERROR! Wrong initial condition type: %d\n This should never happen!\n", initcond_type);
      exit(1);
  }

  return initID;
}