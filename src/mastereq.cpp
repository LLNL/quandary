#include "mastereq.hpp"




MasterEq::MasterEq(){
  dim = 0;
  oscil_vec = NULL;
  RHS = NULL;
  Ad     = NULL;
  Bd     = NULL;
  Ac_vec = NULL;
  Bc_vec = NULL;
  dRedp = NULL;
  dImdp = NULL;
  usematfree = false;
}


MasterEq::MasterEq(std::vector<int> nlevels_, Oscillator** oscil_vec_, const std::vector<double> xi_, LindbladType lindbladtype, const std::vector<double> collapse_time_, bool usematfree_) {
  int ierr;

  nlevels = nlevels_;
  noscillators = nlevels.size();
  oscil_vec = oscil_vec_;
  xi = xi_;
  collapse_time = collapse_time_;
  assert(xi.size() >= (noscillators+1) * noscillators / 2);
  assert(collapse_time.size() >= 2*noscillators);
  usematfree = usematfree_;

  int mpisize_petsc;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_petsc);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);

  /* Dimension of vectorized system: (n_1*...*n_q)^2 */
  dim = 1;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    dim *= oscil_vec[iosc]->getNLevels();
  }
  int dimmat = dim;
  dim = dim*dim; // density matrix: N \times N -> vectorized: N^2

  /* Sanity check for parallel petsc */
  if (dim % mpisize_petsc != 0) {
    printf("\n ERROR in parallel distribution: Petsc's communicator size (%d) must be integer multiple of system dimension N^2=%d\n", mpisize_petsc, dim);
    exit(1);
  }

  /* Create matrix shell for applying system matrix (RHS), */
  /* dimension: 2*dim x 2*dim for the real-valued system */
  MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2*dim, 2*dim, (void**) &RHSctx, &RHS);
  MatSetOptionsPrefix(RHS, "system");
  MatSetFromOptions(RHS); MatSetUp(RHS);
  MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY);

  /* Allocate time-varying building blocks */
  Mat loweringOP, loweringOP_T;
  Mat numberOP;
  Ac_vec = new Mat[noscillators];
  Bc_vec = new Mat[noscillators];

  /* Compute building blocks */
  for (int iosc = 0; iosc < noscillators; iosc++) {

    /* Get lowering operator a = I_(n_1) \kron ... \kron a^(n_k) \kron ... \kron I_(n_q) */
    loweringOP = oscil_vec[iosc]->getLoweringOP((bool)mpirank_petsc);
    MatTranspose(loweringOP, MAT_INITIAL_MATRIX, &loweringOP_T);

    /* Compute Ac = I_N \kron (a - a^T) - (a - a^T) \kron I_N */
    MatCreate(PETSC_COMM_WORLD, &Ac_vec[iosc]);
    MatSetSizes(Ac_vec[iosc], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatSetUp(Ac_vec[iosc]);
    MatSetFromOptions(Ac_vec[iosc]);
    Ikron(loweringOP,   dimmat,  1.0, &Ac_vec[iosc], ADD_VALUES);
    Ikron(loweringOP_T, dimmat, -1.0, &Ac_vec[iosc], ADD_VALUES);
    kronI(loweringOP_T, dimmat, -1.0, &Ac_vec[iosc], ADD_VALUES);
    kronI(loweringOP,   dimmat,  1.0, &Ac_vec[iosc], ADD_VALUES);
    MatAssemblyBegin(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    
    /* Compute Bc = - I_N \kron (a + a^T) + (a + a^T) \kron I_N */
    MatCreate(PETSC_COMM_WORLD, &Bc_vec[iosc]);
    MatSetSizes(Bc_vec[iosc], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatSetUp(Bc_vec[iosc]);
    MatSetFromOptions(Bc_vec[iosc]);
    Ikron(loweringOP,   dimmat, -1.0, &Bc_vec[iosc], ADD_VALUES);
    Ikron(loweringOP_T, dimmat, -1.0, &Bc_vec[iosc], ADD_VALUES);
    kronI(loweringOP_T, dimmat,  1.0, &Bc_vec[iosc], ADD_VALUES);
    kronI(loweringOP,   dimmat,  1.0, &Bc_vec[iosc], ADD_VALUES);
    MatAssemblyBegin(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);

    MatDestroy(&loweringOP_T);
  }

  /* Allocate and compute imag drift part Bd = Hd */
  MatCreate(PETSC_COMM_WORLD, &Bd);
  MatSetSizes(Bd, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatSetUp(Bd);
  MatSetFromOptions(Bd);
  int xi_id = 0;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    Mat tmp, tmp_T;
    Mat numberOPj;
    
    /* Get the number operator */
    // Zero mat on all but the first petsc procs */
    numberOP = oscil_vec[iosc]->getNumberOP((bool) mpirank_petsc);

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
    for (int josc = iosc+1; josc < noscillators; josc++) {
      numberOPj = oscil_vec[josc]->getNumberOP((bool) mpirank_petsc);
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
  MatSetFromOptions(Ad);
  MatSetUp(Ad);
  for (int iosc = 0; iosc < noscillators; iosc++) {
  
    switch (lindbladtype)  {
      case NONE:
        continue;
        break;
      case DECAY: 
        L1 = oscil_vec[iosc]->getLoweringOP((bool)mpirank_petsc);
        addT1 = true;
        addT2 = false;
        break;
      case DEPHASE:
        L2 = oscil_vec[iosc]->getNumberOP((bool)mpirank_petsc);
        addT1 = false;
        addT2 = true;
        break;
      case BOTH:
        L1 = oscil_vec[iosc]->getLoweringOP((bool)mpirank_petsc);
        L2 = oscil_vec[iosc]->getNumberOP((bool)mpirank_petsc);
        addT1 = true;
        addT2 = true;
        break;
      default:
        printf("ERROR! Wrong lindblad type: %d\n", lindbladtype);
        exit(1);
    }

    /* --- Adding T1-DECAY (L1 = a_j) for oscillator j --- */
    if (addT1 && collapse_time[iosc*2] > 1e-14) { 
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
    if (addT2 && collapse_time[iosc*2+1] > 1e-14) { 
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

  /* Create vector strides for accessing Re and Im part in x */
  int ilow, iupp;
  MatGetOwnershipRange(RHS, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  /* Compute maximum number of design parameters over all oscillators */
  nparams_max = 0;
  for (int ioscil = 0; ioscil < getNOscillators(); ioscil++) {
      int n = getOscillator(ioscil)->getNParams(); 
      if (n > nparams_max) nparams_max = n;
  }

  /* Allocate some auxiliary vectors */
  dRedp = new double[nparams_max];
  dImdp = new double[nparams_max];
  cols = new int[nparams_max];
  vals = new double [nparams_max];
  
  MatCreateVecs(Ac_vec[0], &Acu, NULL);
  MatCreateVecs(Ac_vec[0], &Acv, NULL);
  MatCreateVecs(Bc_vec[0], &Bcu, NULL);
  MatCreateVecs(Bc_vec[0], &Bcv, NULL);
  MatCreateVecs(Bc_vec[0], &auxil, NULL);

  /* Allocate MatShell context for applying RHS */
  RHSctx.isu = &isu;
  RHSctx.isv = &isv;
  RHSctx.xi = xi;
  RHSctx.collapse_time = collapse_time;
  RHSctx.Ac_vec = &Ac_vec;
  RHSctx.Bc_vec = &Bc_vec;
  RHSctx.Ad = &Ad;
  RHSctx.Bd = &Bd;
  RHSctx.Acu = &Acu;
  RHSctx.Acv = &Acv;
  RHSctx.Bcu = &Bcu;
  RHSctx.Bcv = &Bcv;
  RHSctx.nlevels = nlevels;
  RHSctx.oscil_vec = &oscil_vec;
  RHSctx.time = 0.0;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    RHSctx.control_Re.push_back(0.0);
    RHSctx.control_Im.push_back(0.0);
  }

  /* Set the MatMult routine for applying the RHS to a vector x */
  if (usematfree) MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMultMatFree_2Osc);
  else            MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMultSparseMat);
  MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTransposeSparseMat);

}


MasterEq::~MasterEq(){
  if (dim > 0){
    MatDestroy(&RHS);
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
    // delete [] vals;

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

int MasterEq::assemble_RHS(double t){
  int ierr;

  /* Prepare the matrix shell to perform the action of RHS on a vector */
  RHSctx.time = t;

  for (int iosc = 0; iosc < noscillators; iosc++) {
    double p, q;
    oscil_vec[iosc]->evalControl(t, &p, &q);
    RHSctx.control_Re[iosc] = p;
    RHSctx.control_Im[iosc] = q;
  }

  return 0;
}



Mat MasterEq::getRHS() { return RHS; }


void MasterEq::createReducedDensity(Vec rho, Vec *reduced, std::vector<int>oscilIDs) {

  Vec red;

  /* Get dimensions of preceding and following subsystem */
  int dim_pre  = 1; 
  int dim_post = 1;
  for (int iosc = 0; iosc < oscilIDs[0]; iosc++) 
    dim_pre  *= getOscillator(iosc)->getNLevels();
  for (int iosc = oscilIDs[oscilIDs.size()-1]+1; iosc < getNOscillators(); iosc++) 
    dim_post *= getOscillator(iosc)->getNLevels();

  int dim_reduced = 1;
  for (int i = 0; i < oscilIDs.size();i++) {
    dim_reduced *= getOscillator(oscilIDs[i])->getNLevels();
  }

  /* sanity test */
  int dimmat = dim_pre * dim_reduced * dim_post;
  assert ( (int) pow(dimmat,2) == dim);

  /* Get local ownership of incoming full density matrix */
  int ilow, iupp;
  VecGetOwnershipRange(rho, &ilow, &iupp);

  /* Create reduced density matrix, sequential */
  VecCreateSeq(PETSC_COMM_SELF, 2*dim_reduced*dim_reduced, &red);
  VecSetFromOptions(red);

  /* Iterate over reduced density matrix elements */
  for (int i=0; i<dim_reduced; i++) {
    for (int j=0; j<dim_reduced; j++) {
      double sum_re = 0.0;
      double sum_im = 0.0;
      /* Iterate over all dim_pre blocks of size n_k * dim_post */
      for (int l = 0; l < dim_pre; l++) {
        int blockstartID = l * dim_reduced * dim_post; // Go to beginning of block 
        /* iterate over elements in this block */
        for (int m=0; m<dim_post; m++) {
          int rho_row = blockstartID + i * dim_post + m;
          int rho_col = blockstartID + j * dim_post + m;
          int rho_vecID_re = getIndexReal(rho_col * dimmat + rho_row);
          int rho_vecID_im = getIndexImag(rho_col * dimmat + rho_row);
          /* Get real and imaginary part from full density matrix */
          double re = 0.0;
          double im = 0.0;
          if (ilow <= rho_vecID_re && rho_vecID_re < iupp) {
            VecGetValues(rho, 1, &rho_vecID_re, &re);
            VecGetValues(rho, 1, &rho_vecID_im, &im);
          } 
          /* add to partial trace */
          sum_re += re;
          sum_im += im;
        }
      }
      /* Set real and imaginary part of element (i,j) of the reduced density matrix */
      int out_vecID_re = getIndexReal(j * dim_reduced + i);
      int out_vecID_im = getIndexImag(j * dim_reduced + i);
      VecSetValues( red, 1, &out_vecID_re, &sum_re, INSERT_VALUES);
      VecSetValues( red, 1, &out_vecID_im, &sum_im, INSERT_VALUES);
    }
  }
  VecAssemblyBegin(red);
  VecAssemblyEnd(red);

  /* Sum up from all petsc cores. This is not at all a good solution. TODO: Change this! */
  double* dataptr;
  int size = 2*dim_reduced*dim_reduced;
  double* mydata = new double[size];
  VecGetArray(red, &dataptr);
  for (int i=0; i<size; i++) {
    mydata[i] = dataptr[i];
  }
  MPI_Allreduce(mydata, dataptr, size, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  VecRestoreArray(red, &dataptr);
  delete [] mydata;

  /* Set output */
  *reduced = red;
}


void MasterEq::createReducedDensity_diff(Vec rhobar, Vec reducedbar, std::vector<int> oscilIDs) {
  
  /* Get dimensions of preceding and following subsystem */
  int dim_pre  = 1; 
  int dim_post = 1;
  for (int iosc = 0; iosc < oscilIDs[0]; iosc++) 
    dim_pre  *= getOscillator(iosc)->getNLevels();
  for (int iosc = oscilIDs[oscilIDs.size()-1]+1; iosc < getNOscillators(); iosc++) 
    dim_post *= getOscillator(iosc)->getNLevels();

  int dim_reduced = 1;
  for (int i = 0; i < oscilIDs.size();i++) {
    dim_reduced *= getOscillator(oscilIDs[i])->getNLevels();
  }

  /* Get local ownership of full density rhobar */
  int ilow, iupp;
  VecGetOwnershipRange(rhobar, &ilow, &iupp);

  /* Get local ownership of reduced density bar*/
  int ilow_red, iupp_red;
  VecGetOwnershipRange(reducedbar, &ilow_red, &iupp_red);

  /* sanity test */
  int dimmat = dim_pre * dim_reduced * dim_post;
  assert ( (int) pow(dimmat,2) == dim);

 /* Iterate over reduced density matrix elements */
  for (int i=0; i<dim_reduced; i++) {
    for (int j=0; j<dim_reduced; j++) {
      /* Get value from reducedbar */
      int vecID_re = getIndexReal(j * dim_reduced + i);
      int vecID_im = getIndexImag(j * dim_reduced + i);
      double re = 0.0;
      double im = 0.0;
      VecGetValues( reducedbar, 1, &vecID_re, &re);
      VecGetValues( reducedbar, 1, &vecID_im, &im);

      /* Iterate over all dim_pre blocks of size n_k * dim_post */
      for (int l = 0; l < dim_pre; l++) {
        int blockstartID = l * dim_reduced * dim_post; // Go to beginning of block 
        /* iterate over elements in this block */
        for (int m=0; m<dim_post; m++) {
          /* Set values into rhobar */
          int rho_row = blockstartID + i * dim_post + m;
          int rho_col = blockstartID + j * dim_post + m;
          int rho_vecID_re = getIndexReal(rho_col * dimmat + rho_row);
          int rho_vecID_im = getIndexImag(rho_col * dimmat + rho_row);

          /* Set derivative */
          if (ilow <= rho_vecID_re && rho_vecID_re < iupp) {
            VecSetValues(rhobar, 1, &rho_vecID_re, &re, ADD_VALUES);
            VecSetValues(rhobar, 1, &rho_vecID_im, &im, ADD_VALUES);
          }
        }
      }
    }
  }
  VecAssemblyBegin(rhobar); VecAssemblyEnd(rhobar);

}

/* grad += RHS(x)^T * xbar  */
void MasterEq::computedRHSdp(double t, Vec x, Vec xbar, double alpha, Vec grad) {

  /* Get real and imaginary part from x and x_bar */
  Vec u, v, ubar, vbar;
  VecGetSubVector(x, isu, &u);
  VecGetSubVector(x, isv, &v);
  VecGetSubVector(xbar, isu, &ubar);
  VecGetSubVector(xbar, isv, &vbar);

  /* Loop over oscillators */
  int col_shift = 0;
  for (int iosc= 0; iosc < noscillators; iosc++){

    /* Evaluate the derivative of the control functions wrt control parameters */
    for (int i=0; i<nparams_max; i++){
      dRedp[i] = 0.0;
      dImdp[i] = 0.0;
    }
    oscil_vec[iosc]->evalControl_diff(t, dRedp, dImdp);

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

    /* Number of parameters for this oscillator */
    int nparams_iosc = getOscillator(iosc)->getNParams();

    /* Set gradient terms for each control parameter */
    for (int iparam=0; iparam < nparams_iosc; iparam++) {
      vals[iparam] = (uAubar + vAvbar) * dImdp[iparam] + ( -vBubar + uBvbar) * dRedp[iparam];
      cols[iparam] = col_shift + iparam;
    }
    VecSetValues(grad, nparams_iosc, cols, vals, ADD_VALUES);
    col_shift += nparams_iosc;
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


int MasterEq::getRhoT0(int iinit, int ninit, InitialConditionType initcond_type, std::vector<int> oscilIDs, Vec rho0){

  int ilow, iupp;
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
      /* Position in vectorized q(0) */
      row = getIndexReal(diagelem * dim_rho + diagelem);

      /* Assemble */
      VecGetOwnershipRange(rho0, &ilow, &iupp);
      if (ilow <= row && row < iupp) VecSetValue(rho0, row, 1.0, INSERT_VALUES); 
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);

      /* Set initial conditon ID */
      initID = iinit * ninit + iinit;

      break;

    case BASIS:

      /* Reset the initial conditions */
      VecZeroEntries(rho0); 

      /* Get distribution */
      VecGetOwnershipRange(rho0, &ilow, &iupp);

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
        elemID = getIndexReal(elemID); // real part
        double val = 1.0;
        if (ilow <= elemID && elemID < iupp) VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
      } else {
      //   /* B_{kj} contains four non-zeros, two per row */
        int* rows = new int[4];
        double* vals = new double[4];

        /* Get storage index of Re(x) */
        rows[0] = getIndexReal(k * dim_post * dim_rho + k * dim_post); // (k,k)
        rows[1] = getIndexReal(j * dim_post * dim_rho + j * dim_post); // (j,j)
        rows[2] = getIndexReal(j * dim_post * dim_rho + k * dim_post); // (k,j)
        rows[3] = getIndexReal(k * dim_post * dim_rho + j * dim_post); // (j,k)

        if (k < j) { // B_{kj} = 1/2(E_kk + E_jj) + 1/2(E_kj + E_jk)
          vals[0] = 0.5;
          vals[1] = 0.5;
          vals[2] = 0.5;
          vals[3] = 0.5;
          for (int i=0; i<4; i++) {
            if (ilow <= rows[i] && rows[i] < iupp) VecSetValues(rho0, 1, &(rows[i]), &(vals[i]), INSERT_VALUES);
          }
        } else {  // B_{kj} = 1/2(E_kk + E_jj) + i/2(E_jk - E_kj)
          vals[0] = 0.5;
          vals[1] = 0.5;
          for (int i=0; i<2; i++) {
            if (ilow <= rows[i] && rows[i] < iupp) VecSetValues(rho0, 1, &(rows[i]), &(vals[i]), INSERT_VALUES);
          }
          vals[2] = -0.5;
          vals[3] = 0.5;
          rows[2] = getIndexImag(j * dim_post * dim_rho + k * dim_post); // Index of Im(x)
          rows[3] = getIndexImag(k * dim_post * dim_rho + j * dim_post); // Index of Im(x)
          for (int i=2; i<4; i++) {
            if (ilow <= rows[i] && rows[i] < iupp) VecSetValues(rho0, 1, &(rows[i]), &(vals[i]), INSERT_VALUES);
          }
        }
        delete [] rows; 
        delete [] vals;
      }
      
      /* Assemble rho0 */
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);

      /* Set initial condition ID */
      initID = j * ( (int) sqrt(ninit)) + k;

      break;

    default:
      printf("ERROR! Wrong initial condition type: %d\n This should never happen!\n", initcond_type);
      exit(1);
  }

  return initID;
}


/* Define the action of RHS on a vector x */
int myMatMultSparseMat(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  
/* Get u, v from x and y  */
  Vec u, v;
  Vec uout, vout;
  VecGetSubVector(x, *shellctx->isu, &u);
  VecGetSubVector(x, *shellctx->isv, &v);
  VecGetSubVector(y, *shellctx->isu, &uout);
  VecGetSubVector(y, *shellctx->isv, &vout);

  // uout = Re*u - Im*v
  //      = (Ad + sum_k q_kA_k)*u - (Bd + sum_k p_kB_k)*v
  // vout = Im*u + Re*v
  //      = (Bd + sum_k p_kB_k)*u + (Ad + sum_k q_kA_k)*v

  // Constant part uout = Adu - Bdv
  MatMult(*shellctx->Bd, v, uout);
  VecScale(uout, -1.0);
  MatMultAdd(*shellctx->Ad, u, uout, uout);
  // Constant part vout = Adv + Bdu
  MatMult(*shellctx->Ad, v, vout);
  MatMultAdd(*shellctx->Bd, u, vout, vout);

  /* Control part */
  for (int iosc = 0; iosc < shellctx->nlevels.size(); iosc++) {
    /* Get controls */
    double p = shellctx->control_Re[iosc];
    double q = shellctx->control_Im[iosc];

    // uout += q^k*Acu 
    MatMult((*(shellctx->Ac_vec))[iosc], u, *shellctx->Acu);
    VecAXPY(uout, q, *shellctx->Acu);
    // uout -= p^kBcv
    MatMult((*(shellctx->Bc_vec))[iosc], v, *shellctx->Bcv);
    VecAXPY(uout, -1.*p, *shellctx->Bcv);
    // vout += q^kAcv
    MatMult((*(shellctx->Ac_vec))[iosc], v, *shellctx->Acv);
    VecAXPY(vout, q, *shellctx->Acv);
    // vout += p^kBcu
    MatMult((*(shellctx->Bc_vec))[iosc], u, *shellctx->Bcu);
    VecAXPY(vout, p, *shellctx->Bcu);
  }


  /* Restore */
  VecRestoreSubVector(x, *shellctx->isu, &u);
  VecRestoreSubVector(x, *shellctx->isv, &v);
  VecRestoreSubVector(y, *shellctx->isu, &uout);
  VecRestoreSubVector(y, *shellctx->isv, &vout);

  return 0;
}



/* Define the action of RHS on a vector x */
template <int n0, int n1>
int myMatMultMatFree_splitloops(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);


  /* Evaluate coefficients */
  double xi0  = shellctx->xi[0];
  double xi01 = shellctx->xi[1];
  double xi1  = shellctx->xi[2];
  double decay0 = 0.0;
  double decay1 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  if (shellctx->collapse_time[0] > 1e-14)
    decay0 = 1./shellctx->collapse_time[0];
  if (shellctx->collapse_time[1] > 1e-14)
    dephase0 = 1./shellctx->collapse_time[1];
  if (shellctx->collapse_time[2] > 1e-14)
    decay1= 1./shellctx->collapse_time[2];
  if (shellctx->collapse_time[3] > 1e-14)
    dephase1 = 1./shellctx->collapse_time[3];

  /* Diagonal elements: Hd, Dephasing L2, Decay L1 diagonal part*/
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          /* Get output index in vectorized, colocated y */
          int rowre = 2 * it;
          int rowim = rowre + 1;
          
          // Constant Hd part: uout = ( hd(ik) - hd(ik'))*vin
          //                   vout = (-hd(ik) + hd(ik'))*uin
          double hd  = Hd(xi0, xi01, xi1, i0, i1); 
          double hdp = Hd(xi0, xi01, xi1, i0p, i1p); 
          double resre = ( hd - hdp ) * xptr[rowim];
          double resim = (-hd + hdp ) * xptr[rowre];
          // Decay l1, diagonal part: xout += l1diag xin
          // Dephasing l2: xout += l2(ik, ikp) xin
          double l1diag = L1diag(decay0, decay1, i0, i1, i0p, i1p);
          double l2 = L2(dephase0, dephase1, i0, i1, i0p, i1p);
          resre += (l2 + l1diag) * xptr[rowre];
          resim += (l2 + l1diag) * xptr[rowim];

          /* Update */
          yptr[rowre] = resre;
          yptr[rowim] = resim;
          it++;
        }
      }
    }
  }

  /* Decay L1, off-diagonal */
  /* Oscillators 0 */
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i0 < n0-1 && i0p < n0-1) {
            /* Get output index in vectorized, colocated output y */
            int rowre = 2 * it;
            int rowim = rowre + 1;

            double l1off = decay0 * sqrt((i0+1)*(i0p+1));
            int itx = TensorGetIndex(n0,n1,i0+1,i1,i0p+1,i1p);
            int row_xre = 2*(itx);
            int row_xim = row_xre+1;
            yptr[rowre] += l1off * xptr[row_xre];
            yptr[rowim] += l1off * xptr[row_xim];
          }
          it++;
        }
      }
    }
  }
  it = 0;
  /* Oscillators 1 */
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i1 < n1-1 && i1p < n1-1) {
            /* Get output index in vectorized, colocated output y */
            int rowre = 2*(it);
            int rowim = rowre+1;

            double l1off = decay1 * sqrt((i1+1)*(i1p+1));
            int itx = TensorGetIndex(n0,n1,i0,i1+1,i0p,i1p+1);
            int row_xre = 2*(itx);
            int row_xim = row_xre+1;
            yptr[rowre] += l1off * xptr[row_xre];
            yptr[rowim] += l1off * xptr[row_xim];
          }
          it++;
        }
      }
    }
  }


  /* --- Control hamiltonian --- */
  /* Oscil k=0 */
  double pt = shellctx->control_Re[0];
  double qt = shellctx->control_Im[0];
  it=0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i0 < n0-1) {
            /* Get output index in vectorized, colocated y */
            int rowre = 2*(it);
            int rowim = rowre+1;

            /* \rho(ik+1..,ik'..) term */
            int itx = TensorGetIndex(n0, n1,i0+1,i1,i0p,i1p);
            int row_xre = 2*(itx);
            int row_xim = row_xre+1;
            double sq = sqrt(i0 + 1);
            yptr[rowre] += sq * (   pt * xptr[rowim] + qt * xptr[rowre]);
            yptr[rowim] += sq * ( - pt * xptr[rowre] + qt * xptr[rowim]);
          }
          it++;
        }
      }
    }
  }
  it=0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i0p < n0-1) {
            /* Get output index in vectorized, colocated y */
            int rowre = 2*(it);
            int rowim = rowre+1;
            /* \rho(ik..,ik'+1..) */
            double sq = sqrt(i0p + 1);
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p+1,i1p);
            int row_xre = 2*(itx);
            int row_xim = row_xre + 1;
            yptr[rowre] += sq * ( -pt * xptr[row_xim] + qt * xptr[row_xre]);
            yptr[rowim] += sq * (  pt * xptr[row_xre] + qt * xptr[row_xim]);
          }
          it++;
        }
      }
    }
  }
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i0 > 0) {
            /* Get output index in vectorized, colocated y */
            int rowre = 2*(it);
            int rowim = rowre+1;
            /* \rho(ik-1..,ik'..) */
            int itx = TensorGetIndex(n0,n1,i0-1,i1,i0p,i1p);
            int row_xre = 2*itx;
            int row_xim = row_xre+1;
            double sq = sqrt(i0);
            yptr[rowre] += sq * (  pt * xptr[row_xim] - qt * xptr[row_xre]);
            yptr[rowim] += sq * (- pt * xptr[row_xre] - qt * xptr[row_xim]);
          }
          it++;
        }
      }
    }
  }
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i0p > 0) {
          /* Get output index in vectorized, colocated y */
          int rowre = 2*(it);
          int rowim = rowre + 1;
          /* \rho(ik..,ik'-1..) */
          int itx = TensorGetIndex(n0,n1,i0,i1,i0p-1,i1p);
          int row_xre = 2*(itx);
          int row_xim = row_xre + 1;
          double sq = sqrt(i0p);
          yptr[rowre] += sq * (- pt * xptr[row_xim] - qt * xptr[row_xre]);
          yptr[rowim] += sq * (  pt * xptr[row_xre] - qt * xptr[row_xim]);
          }
          it++;
        }
      }
    }
  }

  /* --- Oscil k=1 --- */
  pt = shellctx->control_Re[1];
  qt = shellctx->control_Im[1];
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i1 < n1-1) {
            /* Get output index in vectorized, colocated y */
            int rowre = getIndexReal(it);
            int rowim = getIndexImag(it);
            /* \rho(ik+1..,ik'..) term */
            int itx = TensorGetIndex(n0,n1,i0,i1+1,i0p,i1p);
            int row_xre = getIndexReal(itx);
            int row_xim = getIndexImag(itx);
            double sq = sqrt(i1 + 1);
            yptr[rowre] += sq * (   pt * xptr[row_xim] + qt * xptr[row_xre]);
            yptr[rowim] += sq * ( - pt * xptr[row_xre] + qt * xptr[row_xim]);
          }
          it++;
        }
      }
    }
  }
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i1p < n1-1) {
            /* Get output index in vectorized, colocated y */
            int rowre = getIndexReal(it);
            int rowim = getIndexImag(it);
            /* \rho(ik..,ik'+1..) */
            double sq = sqrt(i1p + 1);
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p,i1p+1);
            int row_xre = getIndexReal(itx);
            int row_xim = getIndexImag(itx);
            yptr[rowre] += sq * ( -pt * xptr[row_xim] + qt * xptr[row_xre]);
            yptr[rowim] += sq * (  pt * xptr[row_xre] + qt * xptr[row_xim]);
          }
          it++;
        }
      }
    }
  }
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i1 > 0) {
            /* Get output index in vectorized, colocated y */
            int rowre = 2*(it);
            int rowim = rowre+1;
            /* \rho(ik-1..,ik'..) */
            int itx = TensorGetIndex(n0,n1,i0,i1-1,i0p,i1p);
            int row_xre = 2 * itx;
            int row_xim = row_xre + 1;
            double sq = sqrt(i1);
            yptr[rowre] += sq * (  pt * xptr[row_xim] - qt * xptr[row_xre]);
            yptr[rowim] += sq * (- pt * xptr[row_xre] - qt * xptr[row_xim]);
          }
          it++;
        }
      }
    }
  }
  it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
          if (i1p > 0) {
            /* Get output index in vectorized, colocated y */
            int rowre = 2*(it);
            int rowim = rowre+1;
            /* \rho(ik..,ik'-1..) */
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p,i1p-1);
            int row_xre = 2*itx;
            int row_xim = row_xre + 1;
            double sq = sqrt(i1p);
            yptr[rowre] += sq * (- pt * xptr[row_xre] - qt * xptr[row_xim]);
            yptr[rowim] += sq * (  pt * xptr[row_xim] - qt * xptr[row_xre]);
          }
          it++;
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

    
  return 0;
}

/* Define the action of RHS^T on a vector x */
int myMatMultTransposeSparseMat(Mat RHS, Vec x, Vec y) {
 
  /* Get time from shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get u, v from x and y  */
  Vec u, v;
  Vec uout, vout;
  VecGetSubVector(x, *shellctx->isu, &u);
  VecGetSubVector(x, *shellctx->isv, &v);
  VecGetSubVector(y, *shellctx->isu, &uout);
  VecGetSubVector(y, *shellctx->isv, &vout);

  // uout = Re^T*u + Im^T*v
  //      = (Ad + sum_k q_kA_k)^T*u + (Bd + sum_k p_kB_k)^T*v
  // vout = -Im^T*u + Re^T*v
  //      = -(Bd + sum_k p_kB_k)^T*u + (Ad + sum_k q_kA_k)^T*v

  // Constant part uout = Ad^Tu + Bd^Tv
  MatMultTranspose(*shellctx->Bd, v, uout);
  MatMultTransposeAdd(*shellctx->Ad, u, uout, uout);
  // Constant part vout = -Bd^Tu + Ad^Tv
  MatMultTranspose(*shellctx->Ad, v, vout);
  VecScale(vout, 1.0);
  MatMultTransposeAdd(*shellctx->Bd, u, vout, vout);

  /* Control part */
  for (int iosc = 0; iosc < shellctx->nlevels.size(); iosc++) {
    /* Get controls */
    double p = shellctx->control_Re[iosc];
    double q = shellctx->control_Im[iosc];

    // uout += q^k*Ac^Tu 
    MatMultTranspose((*(shellctx->Ac_vec))[iosc], u, *shellctx->Acu);
    VecAXPY(uout, q, *shellctx->Acu);
    // uout += p^kBc^Tv
    MatMultTranspose((*(shellctx->Bc_vec))[iosc], v, *shellctx->Bcv);
    VecAXPY(uout, p, *shellctx->Bcv);
    // vout += q^kAc^Tv
    MatMultTranspose((*(shellctx->Ac_vec))[iosc], v, *shellctx->Acv);
    VecAXPY(vout, q, *shellctx->Acv);
    // vout -= p^kBc^Tu
    MatMultTranspose((*(shellctx->Bc_vec))[iosc], u, *shellctx->Bcu);
    VecAXPY(vout, -1.*p, *shellctx->Bcu);
  }



  /* Restore */
  VecRestoreSubVector(x, *shellctx->isu, &u);
  VecRestoreSubVector(x, *shellctx->isv, &v);
  VecRestoreSubVector(y, *shellctx->isu, &uout);
  VecRestoreSubVector(y, *shellctx->isv, &vout);

  return 0;
}



double Hd(const double xi0, const double xi01, const double xi1,const int a, const int b) {
  return - xi0*M_PI * a * (a-1) - xi01*M_PI*2 * a * b - xi1*M_PI * b * (b-1); 
}

double L2(double dephase0, double dephase1, const int i0, const int i1, const int i0p, const int i1p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) );
}



double L1diag(double decay0, double decay1, const int i0, const int i1, const int i0p, const int i1p){
  return - decay0 / 2.0 * ( i0 + i0p ) - decay1 / 2.0 * ( i1 + i1p );
}


int TensorGetIndex(const int nlevels0, const int nlevels1,const  int i0, const int i1, int i0p, const int i1p){
  return i0*nlevels1 + i1 + (nlevels0 * nlevels1) * ( i0p * nlevels1 + i1p);
}






/* Define the action of RHS on a vector x */
template <int n0, int n1>
int myMatMultMatFree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);


  /* Evaluate coefficients */
  double xi0  = shellctx->xi[0];
  double xi01 = shellctx->xi[1];
  double xi1  = shellctx->xi[2];
  double decay0 = 0.0;
  double decay1 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  if (shellctx->collapse_time[0] > 1e-14)
    decay0 = 1./shellctx->collapse_time[0];
  if (shellctx->collapse_time[1] > 1e-14)
    dephase0 = 1./shellctx->collapse_time[1];
  if (shellctx->collapse_time[2] > 1e-14)
    decay1= 1./shellctx->collapse_time[2];
  if (shellctx->collapse_time[3] > 1e-14)
    dephase1 = 1./shellctx->collapse_time[3];
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];

  /* Diagonal elements: Hd, Dephasing L2, Decay L1 diagonal part*/
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {
         
          /* --- Diagonal part ---*/
          //Get input x values 
          double xre = xptr[2 * it];
          double xim = xptr[2 * it + 1];
          // Constant Hd part: uout = ( hd(ik) - hd(ik'))*vin
          //                   vout = (-hd(ik) + hd(ik'))*uin
          double hd  = Hd(xi0, xi01, xi1, i0, i1); 
          double hdp = Hd(xi0, xi01, xi1, i0p, i1p); 
          double yre = ( hd - hdp ) * xim;
          double yim = (-hd + hdp ) * xre;
          // Decay l1, diagonal part: xout += l1diag xin
          // Dephasing l2: xout += l2(ik, ikp) xin
          double l1diag = L1diag(decay0, decay1, i0, i1, i0p, i1p);
          double l2 = L2(dephase0, dephase1, i0, i1, i0p, i1p);
          yre += (l2 + l1diag) * xre; 
          yim += (l2 + l1diag) * xim;

          /* --- Offdiagonal part of decay L1 */
          // Oscillators 0 
          if (i0 < n0-1 && i0p < n0-1) {
            double l1off = decay0 * sqrt((i0+1)*(i0p+1));
            int itx = TensorGetIndex(n0,n1,i0+1,i1,i0p+1,i1p);
            int row_xre = 2*itx;
            int row_xim = row_xre+1;
            yre += l1off * xptr[row_xre];
            yim += l1off * xptr[row_xim];         
          }
          // Oscillator 1 
          if (i1 < n1-1 && i1p < n1-1) {
            double l1off = decay1 * sqrt((i1+1)*(i1p+1));
            int itx = TensorGetIndex(n0,n1,i0,i1+1,i0p,i1p+1);
            int row_xre = 2*(itx);
            int row_xim = row_xre+1;
            yre += l1off * xptr[row_xre];
            yim += l1off * xptr[row_xim];
          }

          /* --- Control hamiltonian --- Oscillator 0 --- */
          /* \rho(ik+1..,ik'..) term */
          if (i0 < n0-1) {
            int itx = TensorGetIndex(n0, n1,i0+1,i1,i0p,i1p);
            int row_xre = 2*(itx);
            int row_xim = row_xre+1;
            double sq = sqrt(i0 + 1);
            yre += sq * (   pt0 * xptr[row_xim] + qt0 * xptr[row_xre]);
            yim += sq * ( - pt0 * xptr[row_xre] + qt0 * xptr[row_xim]);
          }
          /* \rho(ik..,ik'+1..) */
          if (i0p < n0-1) {
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p+1,i1p);
            int row_xre = 2*(itx);
            int row_xim = row_xre + 1;
            double sq = sqrt(i0p + 1);
            yre += sq * ( -pt0 * xptr[row_xim] + qt0 * xptr[row_xre]);
            yim += sq * (  pt0 * xptr[row_xre] + qt0 * xptr[row_xim]);
          }
          /* \rho(ik-1..,ik'..) */
          if (i0 > 0) {
            int itx = TensorGetIndex(n0,n1,i0-1,i1,i0p,i1p);
            int row_xre = 2*itx;
            int row_xim = row_xre+1;
            double sq = sqrt(i0);
            yre += sq * (  pt0 * xptr[row_xim] - qt0 * xptr[row_xre]);
            yim += sq * (- pt0 * xptr[row_xre] - qt0 * xptr[row_xim]);
          }
          /* \rho(ik..,ik'-1..) */
          if (i0p > 0) {
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p-1,i1p);
            int row_xre = 2*(itx);
            int row_xim = row_xre + 1;
            double sq = sqrt(i0p);
            yre += sq * (- pt0 * xptr[row_xim] - qt0 * xptr[row_xre]);
            yim += sq * (  pt0 * xptr[row_xre] - qt0 * xptr[row_xim]);
          }
 
          /* --- Control hamiltonian --- Oscillator 1 --- */
          /* \rho(ik+1..,ik'..) term */
          if (i1 < n1-1) {
            int itx = TensorGetIndex(n0,n1,i0,i1+1,i0p,i1p);
            int row_xre = getIndexReal(itx);
            int row_xim = getIndexImag(itx);
            double sq = sqrt(i1 + 1);
            yre += sq * (   pt1 * xptr[row_xim] + qt1 * xptr[row_xre]);
            yim += sq * ( - pt1 * xptr[row_xre] + qt1 * xptr[row_xim]);
          }
          /* \rho(ik..,ik'+1..) */
          if (i1p < n1-1) {
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p,i1p+1);
            int row_xre = getIndexReal(itx);
            int row_xim = getIndexImag(itx);
            double sq = sqrt(i1p + 1);
            yre += sq * ( -pt1 * xptr[row_xim] + qt1 * xptr[row_xre]);
            yim += sq * (  pt1 * xptr[row_xre] + qt1 * xptr[row_xim]);
          }
          /* \rho(ik-1..,ik'..) */
          if (i1 > 0) {
            int itx = TensorGetIndex(n0,n1,i0,i1-1,i0p,i1p);
            int row_xre = 2 * itx;
            int row_xim = row_xre + 1;
            double sq = sqrt(i1);
            yre += sq * (  pt1 * xptr[row_xim] - qt1 * xptr[row_xre]);
            yim += sq * (- pt1 * xptr[row_xre] - qt1 * xptr[row_xim]);
          }
          /* \rho(ik..,ik'-1..) */
          if (i1p > 0) {
            /* Get output index in vectorized, colocated y */
            int itx = TensorGetIndex(n0,n1,i0,i1,i0p,i1p-1);
            int row_xre = 2*itx;
            int row_xim = row_xre + 1;
            double sq = sqrt(i1p);
            yre += sq * (- pt1 * xptr[row_xim] - qt1 * xptr[row_xim]);
            yim += sq * (  pt1 * xptr[row_xim] - qt1 * xptr[row_xim]);
          }
 
          /* Update */
          yptr[2*it]   = yre;
          yptr[2*it+1] = yim;
          it++;
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

    
  return 0;
}


int myMatMultMatFree_2Osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);


  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  if(n0==3 && n1==20)
  {
    return myMatMultMatFree_splitloops<3,20>(RHS, x, y);
    return myMatMultMatFree<3,20>(RHS, x, y);
  } else if(n0==3 && n1==10){
    return myMatMultMatFree_splitloops<3,10>(RHS, x, y);
    return myMatMultMatFree<3,10>(RHS, x, y);
  } else {
    printf("ERROR: Matrix free implementation NOT IMPLEMENTED for cases other than 3x10 or 3x20!\n");
    exit(1);
  }
}