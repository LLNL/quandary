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
  cols = NULL;
  vals = NULL;
  usematfree = false;
}


MasterEq::MasterEq(std::vector<int> nlevels_, std::vector<int> nessential_, Oscillator** oscil_vec_, const std::vector<double> crosskerr_, const std::vector<double> Jkl_, const std::vector<double> eta_, LindbladType lindbladtype, bool usematfree_) {
  int ierr;

  nlevels = nlevels_;
  nessential = nessential_;
  noscillators = nlevels.size();
  oscil_vec = oscil_vec_;
  crosskerr = crosskerr_;
  Jkl = Jkl_;
  eta = eta_;
  usematfree = usematfree_;

  for (int i=0; i<crosskerr.size(); i++){
    crosskerr[i] *= 2.*M_PI;
  }
  for (int i=0; i<Jkl.size(); i++){
    Jkl[i] *= 2.*M_PI;
  }
  for (int i=0; i<eta.size(); i++){
    eta[i] *= 2.*M_PI;
  }

  int mpisize_petsc;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_petsc);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

  /* Get dimensions */
  dim_rho = 1;
  dim_ess = 1;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    dim_rho *= oscil_vec[iosc]->getNLevels();
    dim_ess *= nessential[iosc];
  }
  dim = dim_rho*dim_rho; // density matrix: N \times N -> vectorized: N^2
  if (mpirank_world == 0) printf("System dimension (complex) N^2 = %d\n",dim);

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

  /* Check Lindblad collapse operator configuration */
  switch (lindbladtype)  {
    case LindbladType::NONE:
      break;
    case LindbladType::DECAY: 
      addT1 = true;
      addT2 = false;
      break;
    case LindbladType::DEPHASE:
      addT1 = false;
      addT2 = true;
      break;
    case LindbladType::BOTH:
      addT1 = true;
      addT2 = true;
      break;
    default:
      printf("ERROR! Wrong lindblad type: %d\n", lindbladtype);
      exit(1);
  } 

  if (!usematfree) {
    initSparseMatSolver();
  }

  /* Create vector strides for accessing Re and Im part in x */
  PetscInt ilow, iupp;
  MatGetOwnershipRange(RHS, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  /* initialize vectors neede for gradient computation */
  initGrad();

  /* Allocate MatShell context for applying RHS */
  RHSctx.isu = &isu;
  RHSctx.isv = &isv;
  RHSctx.crosskerr = crosskerr;
  RHSctx.Jkl = Jkl;
  RHSctx.eta = eta;
  RHSctx.addT1 = addT1;
  RHSctx.addT2 = addT2;
  if (!usematfree){
    RHSctx.Ac_vec = &Ac_vec;
    RHSctx.Bc_vec = &Bc_vec;
    RHSctx.Ad_vec = &Ad_vec;
    RHSctx.Bd_vec = &Bd_vec;
    RHSctx.Ad = &Ad;
    RHSctx.Bd = &Bd;
    RHSctx.aux = &aux;
  }
  RHSctx.nlevels = nlevels;
  RHSctx.oscil_vec = oscil_vec;
  RHSctx.time = 0.0;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    RHSctx.control_Re.push_back(0.0);
    RHSctx.control_Im.push_back(0.0);
  }

  /* Set the MatMult routine for applying the RHS to a vector x */
  if (usematfree) { // matrix-free solver
    if (noscillators == 2) {
      MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_matfree_2Osc);
      MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_matfree_2Osc);
    } else if (noscillators == 3) {
      MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_matfree_3Osc);
      MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_matfree_3Osc);
    } else if (noscillators == 4) {
      MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_matfree_4Osc);
      MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_matfree_4Osc);
    } else if (noscillators == 5) {
      MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_matfree_5Osc);
      MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_matfree_5Osc);
    } else {
      printf("ERROR. Matfree solver only for 2, 3 or 4 oscillators. This should never happen!\n");
      exit(1);
    }
  }
  else { // sparse-matrix solver
    MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_sparsemat);
    MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_sparsemat);
  }

}


MasterEq::~MasterEq(){
  if (dim > 0){
    MatDestroy(&RHS);
    if (!usematfree){
      MatDestroy(&Ad);
      MatDestroy(&Bd);
      for (int iosc = 0; iosc < noscillators; iosc++) {
        MatDestroy(&Ac_vec[iosc]);
        MatDestroy(&Bc_vec[iosc]);
      }
      for (int i= 0; i < noscillators*(noscillators-1)/2; i++) {
        if (fabs(Jkl[i]) > 1e-12 )  {
          MatDestroy(&Ad_vec[i]);
          MatDestroy(&Bd_vec[i]);
        }
      }
      VecDestroy(&aux);
      delete [] Ac_vec;
      delete [] Bc_vec;
      delete [] Ad_vec;
      delete [] Bd_vec;
    }
    delete [] dRedp;
    delete [] dImdp;
    delete [] vals;
    delete [] cols;

    ISDestroy(&isu);
    ISDestroy(&isv);
  }
}


void MasterEq::initSparseMatSolver(){

  /* Allocate time-varying building blocks */
  // control terms
  Ac_vec = new Mat[noscillators];
  Bc_vec = new Mat[noscillators];
  // coupling terms
  Ad_vec = new Mat[noscillators*(noscillators-1)/2];
  Bd_vec = new Mat[noscillators*(noscillators-1)/2];

  int dimmat = (int) sqrt(dim);

  int id_kl=0;  // index for accessing Ad_kl in Ad_vec
  PetscInt ilow, iupp;
  int r1,r2, r1a, r2a, r1b, r2b;
  int col, col1, col2;
  double val;
  // double val1, val2;

  /* Set up control Hamiltonian building blocks Ac, Bc */
  for (int iosc = 0; iosc < noscillators; iosc++) {

    /* Get dimensions */
    int nk     = oscil_vec[iosc]->getNLevels();
    int nprek  = oscil_vec[iosc]->dim_preOsc;
    int npostk = oscil_vec[iosc]->dim_postOsc;

    /* Compute Ac = I_N \kron (a - a^T) - (a - a^T)^T \kron I_N */
    MatCreate(PETSC_COMM_WORLD, &Ac_vec[iosc]);
    MatSetType(Ac_vec[iosc], MATMPIAIJ);
    MatSetSizes(Ac_vec[iosc], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatMPIAIJSetPreallocation(Ac_vec[iosc], 4, NULL, 4, NULL);
    MatSetUp(Ac_vec[iosc]);
    MatSetFromOptions(Ac_vec[iosc]);
    MatGetOwnershipRange(Ac_vec[iosc], &ilow, &iupp);

    /* Iterate over local rows of Ac_vec */
    for (int row = ilow; row<iupp; row++){
      // I_n \kron A_c 
      col1 = row + npostk;
      col2 = row - npostk;
      r1 = row % dimmat;
      r1 = r1 % (nk*npostk);
      r1 = r1 / npostk;
      if (r1 < nk-1) {
        val = sqrt(r1+1);
        if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc], row, col1, val, ADD_VALUES);
      }
      if (r1 > 0) {
        val = -sqrt(r1);
        if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc], row, col2, val, ADD_VALUES);
      } 
      //- A_c \kron I_N
      col1 = row + npostk*dimmat;
      col2 = row - npostk*dimmat;
      r1 = row % (dimmat * nk * npostk);
      r1 = r1 / (dimmat * npostk);
      if (r1 < nk-1) {
        val =  sqrt(r1+1);
        if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc], row, col1, val, ADD_VALUES);
      }
      if (r1 > 0) {
        val = -sqrt(r1);
        if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc], row, col2, val, ADD_VALUES);
      }   
    }
    MatAssemblyBegin(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ac_vec[iosc], MAT_FINAL_ASSEMBLY);

    /* Compute Bc = - I_N \kron (a + a^T) + (a + a^T)^T \kron I_N */
    MatCreate(PETSC_COMM_WORLD, &Bc_vec[iosc]);
    MatSetType(Bc_vec[iosc], MATMPIAIJ);
    MatSetSizes(Bc_vec[iosc], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatMPIAIJSetPreallocation(Bc_vec[iosc], 4, NULL, 4, NULL);
    MatSetUp(Bc_vec[iosc]);
    MatSetFromOptions(Bc_vec[iosc]);
    MatGetOwnershipRange(Bc_vec[iosc], &ilow, &iupp);
    /* Iterate over local rows of Bc_vec */
    for (int row = ilow; row<iupp; row++){
      // - I_n \kron B_c 
      col1 = row + npostk;
      col2 = row - npostk;
      r1 = row % dimmat;
      r1 = r1 % (nk*npostk);
      r1 = r1 / npostk;
      if (r1 < nk-1) {
        val = -sqrt(r1+1);
        if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc], row, col1, val, ADD_VALUES);
      }
      if (r1 > 0) {
        val = -sqrt(r1);
        if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc], row, col2, val, ADD_VALUES);
      } 
      //+ B_c \kron I_N
      col1 = row + npostk*dimmat;
      col2 = row - npostk*dimmat;
      r1 = row % (dimmat * nk * npostk);
      r1 = r1 / (dimmat * npostk);
      if (r1 < nk-1) {
        val =  sqrt(r1+1);
        if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc], row, col1, val, ADD_VALUES);
      }
      if (r1 > 0) {
        val = sqrt(r1);
        if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc], row, col2, val, ADD_VALUES);
      }   
    }
    MatAssemblyBegin(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bc_vec[iosc], MAT_FINAL_ASSEMBLY);


    /* Compute Jaynes-Cummings coupling building blocks */
    /* Ad_kl(t) =  I_N \kron (ak^Tal − akal^T) − (al^Tak − alak^T) \kron IN */
    /* Bd_kl(t) = -I_N \kron (ak^Tal + akal^T) + (al^Tak + alak_T) \kron IN */
    for (int josc=iosc+1; josc<noscillators; josc++){

      if (fabs(Jkl[id_kl]) > 1e-12) { // only allocate if coefficient is non-zero to save memory.

        /* Allocate Ad_kl, Bd_kl matrices, 4 nonzeros per kl-coupling per row. */
        MatCreate(PETSC_COMM_WORLD, &Ad_vec[id_kl]);
        MatCreate(PETSC_COMM_WORLD, &Bd_vec[id_kl]);
        MatSetType(Ad_vec[id_kl], MATMPIAIJ);
        MatSetType(Bd_vec[id_kl], MATMPIAIJ);
        MatSetSizes(Ad_vec[id_kl], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
        MatSetSizes(Bd_vec[id_kl], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
        MatMPIAIJSetPreallocation(Ad_vec[id_kl], 4, NULL, 4, NULL);
        MatMPIAIJSetPreallocation(Bd_vec[id_kl], 4, NULL, 4, NULL);
        MatSetUp(Ad_vec[id_kl]);
        MatSetUp(Bd_vec[id_kl]);
        MatSetFromOptions(Ad_vec[id_kl]);
        MatSetFromOptions(Bd_vec[id_kl]);
        MatGetOwnershipRange(Ad_vec[id_kl], &ilow, &iupp);

        // Dimensions of joscillator
        int nj     = oscil_vec[josc]->getNLevels();
        int nprej  = oscil_vec[josc]->dim_preOsc;
        int npostj = oscil_vec[josc]->dim_postOsc;


        /* Iterate over local rows of Ad_vec / Bd_vec */
        for (int row = ilow; row<iupp; row++){
          // Add +/- I_N \kron (ak^Tal -/+ akal^T)
          r1 = row % (dimmat / nprek);
          r1a = (int) r1 / npostk;
          r1b = r1 % (nj*npostj);
          r1b = r1b % (nj*npostj);
          r1b = (int) r1b / npostj;
          if (r1a > 0 && r1b < nj-1) {
            val = sqrt(r1a * (r1b+1));
            col = row - npostk + npostj;
             if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col,  val, ADD_VALUES);
             if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, -val, ADD_VALUES);
          }
          if (r1a < nk-1  && r1b > 0) {
            val = sqrt((r1a+1) * r1b);
            col = row + npostk - npostj;
            if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col, -val, ADD_VALUES);
            if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, -val, ADD_VALUES);
          }

          // Add -/+ (al^Tak -/+ alak^T) \kron I
          r1 = row % (dimmat * dimmat / nprek );
          r1a = (int) r1 / (npostk*dimmat);
          r1b = r1 % (npostk*dimmat);
          r1b = r1b % (nj*npostj*dimmat);
          r1b = (int) r1b / (npostj*dimmat);
          if (r1a < nk-1 && r1b > 0) {
            val = sqrt((r1a+1) * r1b);
            col = row + npostk*dimmat - npostj*dimmat;
            if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col, -val, ADD_VALUES);
            if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, +val, ADD_VALUES);
          }
          if (r1a > 0 && r1b < nj-1) {
            val = sqrt(r1a * (r1b+1));
            col = row - npostk*dimmat + npostj*dimmat;
            if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col, val, ADD_VALUES);
            if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, val, ADD_VALUES);
          }
        }
        MatAssemblyBegin(Ad_vec[id_kl], MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(Bd_vec[id_kl], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Ad_vec[id_kl], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Bd_vec[id_kl], MAT_FINAL_ASSEMBLY);
      }
      id_kl++;
    }
  }

  /* Allocate and compute imag drift part Bd = Hd */
  MatCreate(PETSC_COMM_WORLD, &Bd);
  MatSetType(Bd, MATMPIAIJ);
  MatSetSizes(Bd, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatMPIAIJSetPreallocation(Bd, 1, NULL, 1, NULL);
  MatSetUp(Bd);
  MatSetFromOptions(Bd);
  MatGetOwnershipRange(Bd, &ilow, &iupp);
  int coupling_id = 0;
  for (int iosc = 0; iosc < noscillators; iosc++) {

    int nk     = oscil_vec[iosc]->getNLevels();
    int nprek  = oscil_vec[iosc]->dim_preOsc;
    int npostk = oscil_vec[iosc]->dim_postOsc;
    double xik = oscil_vec[iosc]->getSelfkerr();
    double detunek = oscil_vec[iosc]->getDetuning();

    /* Diagonal: detuning and anharmonicity  */
    /* Iterate over local rows of Bd */
    for (int row = ilow; row<iupp; row++){

      // Indices for -I_N \kron B_d
      r1 = row % dimmat;
      r1 = r1 % (nk * npostk);
      r1 = (int) r1 / npostk;
      // Indices for B_d \kron I_N
      r2 = (int) row / dimmat;
      r2 = r2 % (nk * npostk);
      r2 = (int) r2 / npostk;

      // -I_N \kron B_d + B_d \kron I_N
      val  = - ( detunek * r1 - xik / 2. * (r1*r1 - r1) );
      val +=     detunek * r2 - xik / 2. * (r2*r2 - r2)  ;
      if (fabs(val)>1e-14) MatSetValue(Bd, row, row, val, ADD_VALUES);
    }

    /* zz-coupling term  -xi_ij * 2 * PI * (N_i*N_j) for j > i */
    for (int josc = iosc+1; josc < noscillators; josc++) {
      int nj     = oscil_vec[josc]->getNLevels();
      int npostj = oscil_vec[josc]->dim_postOsc;
      double xikj = crosskerr[coupling_id];
      coupling_id++;
        
      for (int row = ilow; row<iupp; row++){
        r1 = row % dimmat;
        r1 = r1 % (nk * npostk);
        r1a = r1 / npostk;
        r1b = r1 % npostk;
        r1b = r1b % (nj*npostj);
        r1b = r1b / npostj;

        r2 = (int) row / dimmat;
        r2 = r2 % (nk * npostk);
        r2a = r2 / npostk;
        r2b = r2 % npostk;
        r2b = r2b % (nj*npostj);
        r2b = r2b / npostj;

        // -I_N \kron B_d + B_d \kron I_N
        val =  xikj * r1a * r1b  - xikj * r2a * r2b;
        if (fabs(val)>1e-14) MatSetValue(Bd, row, row, val, ADD_VALUES);
      }
    }

  }
  MatAssemblyBegin(Bd, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Bd, MAT_FINAL_ASSEMBLY);

  /* Allocate and compute real drift part Ad = Lindblad */
  MatCreate(PETSC_COMM_WORLD, &Ad);
  MatSetSizes(Ad, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  if (addT1 || addT2) { // if Lindblad terms, preallocate matrix. Otherwise, leave zero matrix
    MatSetType(Ad, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Ad, noscillators+5, NULL, noscillators+5, NULL);
  }
  MatSetFromOptions(Ad);
  MatSetUp(Ad);
  MatGetOwnershipRange(Ad, &ilow, &iupp);

  if (addT1 || addT2) {  // leave matrix empty if no T1 or T2 decay
    for (int iosc = 0; iosc < noscillators; iosc++) {

      /* Get T1, T2 times */
      double gammaT1 = 0.0;
      double gammaT2 = 0.0;
      if (oscil_vec[iosc]->getDecayTime()   > 1e-14) gammaT1 = 1./(oscil_vec[iosc]->getDecayTime());
      if (oscil_vec[iosc]->getDephaseTime() > 1e-14) gammaT2 = 1./(oscil_vec[iosc]->getDephaseTime());

      // Dimensions 
      int nk     = oscil_vec[iosc]->getNLevels();
      int nprek  = oscil_vec[iosc]->dim_preOsc;
      int npostk = oscil_vec[iosc]->dim_postOsc;

      /* Iterate over local rows of Ad */
      for (int row = ilow; row<iupp; row++){

        /* Add Ad += gamma_j * L \kron L */
        r1 = row % (dimmat*nk*npostk);
        r1a = r1 / (dimmat*npostk);
        r1b = r1 % (npostk*dimmat);
        r1b = r1b % (nk*npostk);
        r1b = r1b / npostk;
        // T1  decay (L1 = a_j)
        if (addT1) { 
          if (r1a < nk-1 && r1b < nk-1) {
            val = gammaT1 * sqrt( (r1a+1) * (r1b+1) );
            col1 = row + npostk * dimmat + npostk;
            if (fabs(val)>1e-14) MatSetValue(Ad, row, col1, val, ADD_VALUES);
          }
        }
        // T2  dephasing (L1 = a_j^Ta_j)
        if (addT2) { 
          val = gammaT2 * r1a * r1b ;
          if (fabs(val)>1e-14) MatSetValue(Ad, row, row, val, ADD_VALUES);
        }

        /* Add Ad += - gamma_j/2  I_n  \kron L^TL  */
        r1 = row % (nk*npostk);
        r1 = r1 / npostk;
        if (addT1) {
          val = - gammaT1/2. * r1;
          if (fabs(val)>1e-14) MatSetValue(Ad, row, row, val, ADD_VALUES);
        }
        if (addT2) {
          val = -gammaT2/2. * r1*r1;
          if (fabs(val)>1e-14) MatSetValue(Ad, row, row, val, ADD_VALUES);
        }

        /* Add Ad += - gamma_j/2  L^TL \kron I_n */
        r1 = row % (nk*npostk*dimmat);
        r1 = r1 / (npostk*dimmat);
        if (addT1) {
          val = -gammaT1/2. * r1;
          if (fabs(val)>1e-14) MatSetValue(Ad, row, row, val, ADD_VALUES);
        }
        if (addT2) {
          val = -gammaT2/2. * r1*r1;
          if (fabs(val)>1e-14) MatSetValue(Ad, row, row, val, ADD_VALUES);
        }
      }
    }
  }
  MatAssemblyBegin(Ad, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ad, MAT_FINAL_ASSEMBLY);

  /* Allocate some auxiliary vectors */
  MatCreateVecs(Ac_vec[0], &aux, NULL);
}

void MasterEq::initGrad(bool refined){
  /* Compute and store maximum number of design parameters over all oscillators */
  nparams_max = 0;
  for (int ioscil = 0; ioscil < getNOscillators(); ioscil++) {
      int n = getOscillator(ioscil)->getNParams();
      if (n > nparams_max) nparams_max = n;
  }

  /* Allocate some auxiliary vectors */
  if (refined) delete [] dRedp;
  if (refined) delete [] dImdp;
  if (refined) delete [] cols;
  if (refined) delete [] vals;
  dRedp = new double[nparams_max];
  dImdp = new double[nparams_max];
  cols = new PetscInt[nparams_max];
  vals = new PetscScalar[nparams_max];
}

int MasterEq::getDim(){ return dim; }

int MasterEq::getDimEss(){ return dim_ess; }

int MasterEq::getDimRho(){ return dim_rho; }

int MasterEq::getNOscillators() { return noscillators; }

Oscillator* MasterEq::getOscillator(const int i) { return oscil_vec[i]; }

int MasterEq::assemble_RHS(const double t){
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


// void MasterEq::createReducedDensity(const Vec rho, Vec *reduced, const std::vector<int>& oscilIDs) {

//   Vec red;

//   /* Get dimensions of preceding and following subsystem */
//   int dim_pre  = 1; 
//   int dim_post = 1;
//   for (int iosc = 0; iosc < oscilIDs[0]; iosc++) 
//     dim_pre  *= getOscillator(iosc)->getNLevels();
//   for (int iosc = oscilIDs[oscilIDs.size()-1]+1; iosc < getNOscillators(); iosc++) 
//     dim_post *= getOscillator(iosc)->getNLevels();

//   int dim_reduced = 1;
//   for (int i = 0; i < oscilIDs.size();i++) {
//     dim_reduced *= getOscillator(oscilIDs[i])->getNLevels();
//   }

//   /* sanity test */
//   int dimmat = dim_pre * dim_reduced * dim_post;
//   assert ( (int) pow(dimmat,2) == dim);

//   /* Get local ownership of incoming full density matrix */
//   int ilow, iupp;
//   VecGetOwnershipRange(rho, &ilow, &iupp);

//   /* Create reduced density matrix, sequential */
//   VecCreateSeq(PETSC_COMM_SELF, 2*dim_reduced*dim_reduced, &red);
//   VecSetFromOptions(red);

//   /* Iterate over reduced density matrix elements */
//   for (int i=0; i<dim_reduced; i++) {
//     for (int j=0; j<dim_reduced; j++) {
//       double sum_re = 0.0;
//       double sum_im = 0.0;
//       /* Iterate over all dim_pre blocks of size n_k * dim_post */
//       for (int l = 0; l < dim_pre; l++) {
//         int blockstartID = l * dim_reduced * dim_post; // Go to beginning of block 
//         /* iterate over elements in this block */
//         for (int m=0; m<dim_post; m++) {
//           int rho_row = blockstartID + i * dim_post + m;
//           int rho_col = blockstartID + j * dim_post + m;
//           int rho_vecID_re = getIndexReal(getVecID(rho_row, rho_col, dimmat));
//           int rho_vecID_im = getIndexImag(getVecID(rho_row, rho_col, dimmat));
//           /* Get real and imaginary part from full density matrix */
//           double re = 0.0;
//           double im = 0.0;
//           if (ilow <= rho_vecID_re && rho_vecID_re < iupp) {
//             VecGetValues(rho, 1, &rho_vecID_re, &re);
//             VecGetValues(rho, 1, &rho_vecID_im, &im);
//           } 
//           /* add to partial trace */
//           sum_re += re;
//           sum_im += im;
//         }
//       }
//       /* Set real and imaginary part of element (i,j) of the reduced density matrix */
//       int out_vecID_re = getIndexReal(getVecID(i, j, dim_reduced));
//       int out_vecID_im = getIndexImag(getVecID(i, j, dim_reduced));
//       VecSetValues( red, 1, &out_vecID_re, &sum_re, INSERT_VALUES);
//       VecSetValues( red, 1, &out_vecID_im, &sum_im, INSERT_VALUES);
//     }
//   }
//   VecAssemblyBegin(red);
//   VecAssemblyEnd(red);

//   /* Sum up from all petsc cores. This is not at all a good solution. TODO: Change this! */
//   double* dataptr;
//   int size = 2*dim_reduced*dim_reduced;
//   double* mydata = new double[size];
//   VecGetArray(red, &dataptr);
//   for (int i=0; i<size; i++) {
//     mydata[i] = dataptr[i];
//   }
//   MPI_Allreduce(mydata, dataptr, size, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
//   VecRestoreArray(red, &dataptr);
//   delete [] mydata;

//   /* Set output */
//   *reduced = red;
// }


// void MasterEq::createReducedDensity_diff(Vec rhobar, const Vec reducedbar,const std::vector<int>& oscilIDs) {

//   /* Get dimensions of preceding and following subsystem */
//   int dim_pre  = 1; 
//   int dim_post = 1;
//   for (int iosc = 0; iosc < oscilIDs[0]; iosc++) 
//     dim_pre  *= getOscillator(iosc)->getNLevels();
//   for (int iosc = oscilIDs[oscilIDs.size()-1]+1; iosc < getNOscillators(); iosc++) 
//     dim_post *= getOscillator(iosc)->getNLevels();

//   int dim_reduced = 1;
//   for (int i = 0; i < oscilIDs.size();i++) {
//     dim_reduced *= getOscillator(oscilIDs[i])->getNLevels();
//   }

//   /* Get local ownership of full density rhobar */
//   int ilow, iupp;
//   VecGetOwnershipRange(rhobar, &ilow, &iupp);

//   /* Get local ownership of reduced density bar*/
//   int ilow_red, iupp_red;
//   VecGetOwnershipRange(reducedbar, &ilow_red, &iupp_red);

//   /* sanity test */
//   int dimmat = dim_pre * dim_reduced * dim_post;
//   assert ( (int) pow(dimmat,2) == dim);

//  /* Iterate over reduced density matrix elements */
//   for (int i=0; i<dim_reduced; i++) {
//     for (int j=0; j<dim_reduced; j++) {
//       /* Get value from reducedbar */
//       int vecID_re = getIndexReal(getVecID(i, j, dim_reduced));
//       int vecID_im = getIndexImag(getVecID(i, j, dim_reduced));
//       double re = 0.0;
//       double im = 0.0;
//       VecGetValues( reducedbar, 1, &vecID_re, &re);
//       VecGetValues( reducedbar, 1, &vecID_im, &im);

//       /* Iterate over all dim_pre blocks of size n_k * dim_post */
//       for (int l = 0; l < dim_pre; l++) {
//         int blockstartID = l * dim_reduced * dim_post; // Go to beginning of block 
//         /* iterate over elements in this block */
//         for (int m=0; m<dim_post; m++) {
//           /* Set values into rhobar */
//           int rho_row = blockstartID + i * dim_post + m;
//           int rho_col = blockstartID + j * dim_post + m;
//           int rho_vecID_re = getIndexReal(getVecID(rho_row, rho_col, dimmat));
//           int rho_vecID_im = getIndexImag(getVecID(rho_row, rho_col, dimmat));

//           /* Set derivative */
//           if (ilow <= rho_vecID_re && rho_vecID_re < iupp) {
//             VecSetValues(rhobar, 1, &rho_vecID_re, &re, ADD_VALUES);
//             VecSetValues(rhobar, 1, &rho_vecID_im, &im, ADD_VALUES);
//           }
//         }
//       }
//     }
//   }
//   VecAssemblyBegin(rhobar); VecAssemblyEnd(rhobar);

// }

/* grad += alpha * RHS(x)^T * xbar  */
void MasterEq::computedRHSdp(const double t, const Vec x, const Vec xbar, const double alpha, Vec grad) {


  if (usematfree) {  // Matrix-free solver
    double res_p_re,  res_p_im, res_q_re, res_q_im;

    const double* xptr, *xbarptr;
    VecGetArrayRead(x, &xptr);
    VecGetArrayRead(xbar, &xbarptr);

    double* coeff_p = new double [noscillators];
    double* coeff_q = new double [noscillators];
    for (int i=0; i<noscillators; i++){
      coeff_p[i] = 0.0;
      coeff_q[i] = 0.0;
    }

    if (noscillators == 2) {
    /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
      int n0 = nlevels[0];
      int n1 = nlevels[1];
      int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
      int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
      int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
      int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);

      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0; i0p++)  {
        for (int i1p = 0; i1p < n1; i1p++)  {
          for (int i0 = 0; i0 < n0; i0++)  {
            for (int i1 = 0; i1 < n1; i1++)  {
              /* Get xbar */
              double xbarre = xbarptr[2*it];
              double xbarim = xbarptr[2*it+1];

              /* --- Oscillator 0 --- */
              dRHSdp_getcoeffs(it, n0, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
              coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
              coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
              /* --- Oscillator 1 --- */
              dRHSdp_getcoeffs(it, n1, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
              coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
              coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;

              it++;
            }
          }
        }
      }
    } else if (noscillators == 3) {
      /* compute strides for accessing x */
      int n0 = nlevels[0];
      int n1 = nlevels[1];
      int n2 = nlevels[2];
      int stridei0  = TensorGetIndex(n0,n1,n2, 1,0,0,0,0,0);
      int stridei1  = TensorGetIndex(n0,n1,n2, 0,1,0,0,0,0);
      int stridei2  = TensorGetIndex(n0,n1,n2, 0,0,1,0,0,0);
      int stridei0p = TensorGetIndex(n0,n1,n2, 0,0,0,1,0,0);
      int stridei1p = TensorGetIndex(n0,n1,n2, 0,0,0,0,1,0);
      int stridei2p = TensorGetIndex(n0,n1,n2, 0,0,0,0,0,1);

      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0; i0p++)  {
        for (int i1p = 0; i1p < n1; i1p++)  {
          for (int i2p = 0; i2p < n2; i2p++)  {
            for (int i0 = 0; i0 < n0; i0++)  {
              for (int i1 = 0; i1 < n1; i1++)  {
                for (int i2 = 0; i2 < n2; i2++)  {
                  /* Get xbar */
                  double xbarre = xbarptr[2*it];
                  double xbarim = xbarptr[2*it+1];

                  /* --- Oscillator 0 --- */
                  dRHSdp_getcoeffs(it, n0, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                  coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
                  coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
                  /* --- Oscillator 1 --- */
                  dRHSdp_getcoeffs(it, n1, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                  coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
                  coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;
                  /* --- Oscillator 2 --- */
                  dRHSdp_getcoeffs(it, n2, i2, i2p, stridei2, stridei2p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                  coeff_p[2] += res_p_re * xbarre + res_p_im * xbarim;
                  coeff_q[2] += res_q_re * xbarre + res_q_im * xbarim;

                  it++;
                }
              }
            }
          }
        }
      }
    } else if (noscillators == 4) {
      /* compute strides for accessing x */
      int n0 = nlevels[0];
      int n1 = nlevels[1];
      int n2 = nlevels[2];
      int n3 = nlevels[3];
      int stridei0  = TensorGetIndex(n0,n1,n2,n3, 1,0,0,0,0,0,0,0);
      int stridei1  = TensorGetIndex(n0,n1,n2,n3, 0,1,0,0,0,0,0,0);
      int stridei2  = TensorGetIndex(n0,n1,n2,n3, 0,0,1,0,0,0,0,0);
      int stridei3  = TensorGetIndex(n0,n1,n2,n3, 0,0,0,1,0,0,0,0);
      int stridei0p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,1,0,0,0);
      int stridei1p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,1,0,0);
      int stridei2p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,0,1,0);
      int stridei3p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,0,0,1);

      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0; i0p++)  {
        for (int i1p = 0; i1p < n1; i1p++)  {
          for (int i2p = 0; i2p < n2; i2p++)  {
            for (int i3p = 0; i3p < n3; i3p++)  {
              for (int i0 = 0; i0 < n0; i0++)  {
                for (int i1 = 0; i1 < n1; i1++)  {
                  for (int i2 = 0; i2 < n2; i2++)  {
                    for (int i3 = 0; i3 < n3; i3++)  {
                      /* Get xbar */
                      double xbarre = xbarptr[2*it];
                      double xbarim = xbarptr[2*it+1];

                      /* --- Oscillator 0 --- */
                      dRHSdp_getcoeffs(it, n0, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
                      /* --- Oscillator 1 --- */
                      dRHSdp_getcoeffs(it, n1, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;
                      /* --- Oscillator 2 --- */
                      dRHSdp_getcoeffs(it, n2, i2, i2p, stridei2, stridei2p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[2] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[2] += res_q_re * xbarre + res_q_im * xbarim;
                      /* --- Oscillator 3 --- */
                      dRHSdp_getcoeffs(it, n3, i3, i3p, stridei3, stridei3p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[3] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[3] += res_q_re * xbarre + res_q_im * xbarim;

                      it++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else if (noscillators == 5) {
      /* compute strides for accessing x */
      int n0 = nlevels[0];
      int n1 = nlevels[1];
      int n2 = nlevels[2];
      int n3 = nlevels[3];
      int n4 = nlevels[4];
      int stridei0  = TensorGetIndex(n0,n1,n2,n3,n4, 1,0,0,0,0,0,0,0,0,0);
      int stridei1  = TensorGetIndex(n0,n1,n2,n3,n4, 0,1,0,0,0,0,0,0,0,0);
      int stridei2  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,1,0,0,0,0,0,0,0);
      int stridei3  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,1,0,0,0,0,0,0);
      int stridei4  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,1,0,0,0,0,0);
      int stridei0p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,1,0,0,0,0);
      int stridei1p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,1,0,0,0);
      int stridei2p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,1,0,0);
      int stridei3p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,0,1,0);
      int stridei4p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,0,0,1);

      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0; i0p++)  {
        for (int i1p = 0; i1p < n1; i1p++)  {
          for (int i2p = 0; i2p < n2; i2p++)  {
            for (int i3p = 0; i3p < n3; i3p++)  {
              for (int i4p = 0; i4p < n4; i4p++)  {
                for (int i0 = 0; i0 < n0; i0++)  {
                  for (int i1 = 0; i1 < n1; i1++)  {
                    for (int i2 = 0; i2 < n2; i2++)  {
                      for (int i3 = 0; i3 < n3; i3++)  {
                        for (int i4 = 0; i4 < n4; i4++)  {
                          /* Get xbar */
                          double xbarre = xbarptr[2*it];
                          double xbarim = xbarptr[2*it+1];

                          /* --- Oscillator 0 --- */
                          dRHSdp_getcoeffs(it, n0, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 1 --- */
                          dRHSdp_getcoeffs(it, n1, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 2 --- */
                          dRHSdp_getcoeffs(it, n2, i2, i2p, stridei2, stridei2p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[2] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[2] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 3 --- */
                          dRHSdp_getcoeffs(it, n3, i3, i3p, stridei3, stridei3p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[3] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[3] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 4 --- */
                          dRHSdp_getcoeffs(it, n4, i4, i4p, stridei4, stridei4p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[4] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[4] += res_q_re * xbarre + res_q_im * xbarim;

                          it++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else { 
      printf("This should never happen!\n"); 
      exit(1);
    }
    VecRestoreArrayRead(x, &xptr);
    VecRestoreArrayRead(xbar, &xbarptr);

    /* Set the gradient values */
    int shift = 0;
    for (int iosc = 0; iosc < noscillators; iosc++){
      // eval control parameters derivatives
      for (int i=0; i<nparams_max; i++){
        dRedp[i] = 0.0;
        dImdp[i] = 0.0;
      }
      oscil_vec[iosc]->evalControl_diff(t, dRedp, dImdp);

      PetscInt nparam = getOscillator(iosc)->getNParams();
      for (int iparam=0; iparam < nparam; iparam++) {
        vals[iparam] = alpha * (coeff_p[iosc] * dRedp[iparam] + coeff_q[iosc] * dImdp[iparam]);
        cols[iparam] = iparam + shift;
      }
      VecSetValues(grad, nparam, cols, vals, ADD_VALUES);
      shift += nparam;
    }

    //Assemble gradient
    VecAssemblyBegin(grad);
    VecAssemblyEnd(grad);

    delete [] coeff_p;
    delete [] coeff_q;
  } else {  // sparse matrix solver

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

    /* Compute terms in RHS(x)^T xbar */
    double uAubar, vAvbar, vBubar, uBvbar;
    MatMult(Ac_vec[iosc], u, aux);
    VecDot(aux, ubar, &uAubar);
    MatMult(Ac_vec[iosc], v, aux);
    VecDot(aux, vbar, &vAvbar);
    MatMult(Bc_vec[iosc], u, aux);
    VecDot(aux, vbar, &uBvbar);
    MatMult(Bc_vec[iosc], v, aux);
    VecDot(aux, ubar, &vBubar);

    /* Number of parameters for this oscillator */
    int nparams_iosc = getOscillator(iosc)->getNParams();

    /* Set gradient terms for each control parameter */
    for (int iparam=0; iparam < nparams_iosc; iparam++) {
      vals[iparam] = alpha * ((uAubar + vAvbar) * dImdp[iparam] + ( -vBubar + uBvbar) * dRedp[iparam]);
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

}

void MasterEq::setControlAmplitudes(const Vec x) {

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


int MasterEq::getRhoT0(const int iinit, const int ninit, const InitialConditionType initcond_type, const std::vector<int>& oscilIDs, Vec rho0){

  PetscInt ilow, iupp; 
  PetscInt elemID;
  double val;
  int dim_post;
  int initID = 1;    // Output: ID for this initial condition */
  int dim_rho = (int) sqrt(dim); // N

  /* Switch over type of initial condition */
  switch (initcond_type) {

    case InitialConditionType::FROMFILE:
      /* Do nothing. Init cond is already stored */
      break;

    case InitialConditionType::PURE:
      /* Do nothing. Init cond is already stored */
      break;

    case InitialConditionType::ENSEMBLE:
      /* Do nothing. Init cond is already stored */
      break;

    case InitialConditionType::THREESTATES:

      /* Reset the initial conditions */
      VecZeroEntries(rho0);

      /* Get partitioning */
      VecGetOwnershipRange(rho0, &ilow, &iupp);

      /* Set the <iinit>'th initial state */
      if (iinit == 0) {
        initID = 1;

        // 1st initial state: rho(0)_IJ = 2(N-i)/(N(N+1)) Delta_IJ
        // in essential dimensions only. Lift up to full dimensions by inserting 0's

        /* Iterate over diagonal elements of essential-dimension system */
        for (int i_ess= 0; i_ess<dim_ess; i_ess++) {
          int diagelem = i_ess;
          double val = 2.*(dim_ess - i_ess) / (dim_ess * (dim_ess + 1));

          if (dim_ess < dim_rho) diagelem = mapEssToFull(diagelem, nlevels, nessential);
          int diagID = getIndexReal(getVecID(diagelem,diagelem,dim_rho));

          if (ilow <= diagID && diagID < iupp) VecSetValue(rho0, diagID, val, INSERT_VALUES);
        }

      } else if (iinit == 1) {
        initID = 2;
        // 2nd initial state: rho(0)_IJ = 1/Nessential
        // in essential dimensions only. Lift up to full dimensions by inserting 0's
        for (int i_ess  = 0; i_ess <dim_ess; i_ess++) {
          for (int j_ess = 0; j_ess <dim_ess; j_ess++) {
            double val = 1./dim_ess;
            int i_full = i_ess;
            int j_full = j_ess;
            if (dim_ess < dim_rho) {
              i_full = mapEssToFull(i_ess, nlevels, nessential);
              j_full = mapEssToFull(j_ess, nlevels, nessential);
            }
            int index = getIndexReal(getVecID(i_full,j_full,dim_rho));   // Re(rho_ij)
            if (ilow <= index && index < iupp) VecSetValue(rho0, index, val, INSERT_VALUES); 
          }
        }

      } else if (iinit == 2) {
        initID = 3;
        // 3rd initial state: rho(0)_IJ = 1/N Delta_IJ
        // in essential dimensions only. Lift up to full dimensions by inserting 0's

        /* Iterate over diagonal elements */
        for (int i_ess = 0; i_ess <dim_ess; i_ess++) {
          double val = 1./ dim_ess;
          int diagelem = i_ess;
          if (dim_ess < dim_rho) diagelem = mapEssToFull(diagelem, nlevels, nessential);
          int diagID = getIndexReal(getVecID(diagelem,diagelem,dim_rho));
          if (ilow <= diagID && diagID < iupp) VecSetValue(rho0, diagID, val, INSERT_VALUES);
        }

      } else {
        printf("ERROR: Wrong initial condition setting!\n");
        exit(1);
      }

      /* Assemble rho0 */
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);
      break;

    case InitialConditionType::NPLUSONE:
      VecGetOwnershipRange(rho0, &ilow, &iupp);

      if (iinit < dim_ess) {
        // First N elements are the Diagonal e_j e_j^\dag.
        // In essential dimensions. Lift up to full dimensions by inserting 0's. 
        VecZeroEntries(rho0);
        elemID = iinit;
        if (dim_ess < dim_rho) elemID = mapEssToFull(elemID, nlevels, nessential);
        elemID = getIndexReal(getVecID(elemID, elemID, dim_rho));
        val = 1.0;
        if (ilow <= elemID && elemID < iupp) VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
        VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);
      }
      else if (iinit == dim_ess) { 
        // fully rotated 1/Ness*Ones(Ness)
        // In essential dimensions. Lift up to full dimensions by inserting 0's. 
        for (int i=0; i<dim_ess; i++){
          for (int j=0; j<dim_ess; j++){
            val = 1.0 / dim_ess;
            int i_full = i;
            int j_full = j;
            if (dim_ess < dim_rho) i_full = mapEssToFull(i, nlevels, nessential);
            if (dim_ess < dim_rho) j_full = mapEssToFull(j, nlevels, nessential);
            elemID = getIndexReal(getVecID(i_full,j_full,dim_rho));
            if (ilow <= elemID && elemID < iupp) VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
            VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);
          }
        }
      }
      else {
        printf("Wrong initial condition index. Should never happen!\n");
        exit(1);
      }
      initID = iinit;

      break;


    case InitialConditionType::DIAGONAL:
      int row, diagelem;

      /* Reset the initial conditions */
      VecZeroEntries(rho0);

      /* Get dimension of partial system behind last oscillator ID (essential levels only) */
      dim_post = 1;
      for (int k = oscilIDs[oscilIDs.size()-1] + 1; k < getNOscillators(); k++) {
        // dim_post *= getOscillator(k)->getNLevels();
        dim_post *= nessential[k];
      }

      /* Compute index of the nonzero element in rho_m(0) = E_pre \otimes |m><m| \otimes E_post */
      diagelem = iinit * dim_post;
      if (dim_ess < dim_rho)  diagelem = mapEssToFull(diagelem, nlevels, nessential);

      /* Set B_{mm} */
      elemID = getIndexReal(getVecID(diagelem, diagelem, dim_rho)); // real part in vectorized system
      val = 1.0;
      VecGetOwnershipRange(rho0, &ilow, &iupp);
      if (ilow <= elemID && elemID < iupp) VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);

      /* Set initial conditon ID */
      initID = iinit * ninit + iinit;

      break;

    case InitialConditionType::BASIS:

      /* Reset the initial conditions */
      VecZeroEntries(rho0);

      /* Get distribution */
      VecGetOwnershipRange(rho0, &ilow, &iupp);

      /* Get dimension of partial system behind last oscillator ID (essential levels only) */
      dim_post = 1;
      for (int k = oscilIDs[oscilIDs.size()-1] + 1; k < getNOscillators(); k++) {
        dim_post *= nessential[k];
      }

      /* Get index (k,j) of basis element B_{k,j} for this initial condition index iinit */
      int k, j;
      k = iinit % ( (int) sqrt(ninit) );
      j = (int) iinit / ( (int) sqrt(ninit) );

      /* Set initial condition ID */
      initID = j * ( (int) sqrt(ninit)) + k;

      /* Set position in rho */
      k = k*dim_post;
      j = j*dim_post;
      if (dim_ess < dim_rho) { 
        k = mapEssToFull(k, nlevels, nessential);
        j = mapEssToFull(j, nlevels, nessential);
      }

      if (k == j) {
        /* B_{kk} = E_{kk} -> set only one element at (k,k) */
        elemID = getIndexReal(getVecID(k, k, dim_rho)); // real part in vectorized system
        double val = 1.0;
        if (ilow <= elemID && elemID < iupp) VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
      } else {
      //   /* B_{kj} contains four non-zeros, two per row */
        PetscInt* rows = new PetscInt[4];
        PetscScalar* vals = new PetscScalar[4];

        /* Get storage index of Re(x) */
        rows[0] = getIndexReal(getVecID(k, k, dim_rho)); // (k,k)
        rows[1] = getIndexReal(getVecID(j, j, dim_rho)); // (j,j)
        rows[2] = getIndexReal(getVecID(k, j, dim_rho)); // (k,j)
        rows[3] = getIndexReal(getVecID(j, k, dim_rho)); // (j,k)

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
          rows[2] = getIndexImag(getVecID(k, j, dim_rho)); // (k,j)
          rows[3] = getIndexImag(getVecID(j, k, dim_rho)); // (j,k)
          for (int i=2; i<4; i++) {
            if (ilow <= rows[i] && rows[i] < iupp) VecSetValues(rho0, 1, &(rows[i]), &(vals[i]), INSERT_VALUES);
          }
        }
        delete [] rows;
        delete [] vals;
      }

      /* Assemble rho0 */
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);

      break;

    default:
      printf("ERROR! Wrong initial condition type: %d\n This should never happen!\n", initcond_type);
      exit(1);
  }

  return initID;
}


/* Sparse matrix solver: Define the action of RHS on a vector x */
int myMatMult_sparsemat(Mat RHS, Vec x, Vec y){

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
  //      = (Ad +  sum_k q_kA_k)*u - (Bd + sum_k p_kB_k)*v
          // + sum_kl J_kl*sin(eta_kl*t) * Ad_kl * u
          //         -J_kl*cos(eta_kl*t) * Bd_kl * v  ]   cross terms
  // vout = Im*u + Re*v
  //      = (Bd + sum_k p_kB_k)*u + (Ad + sum_k q_kA_k)*v
        // + sum_kl J_kl*cos(eta_kl*t) * Bd_kl * u
        //        + J_kl*sin(eta_kl*t) * Ad_kl * v  ]   cross terms

  // Constant part uout = Adu - Bdv
  MatMult(*shellctx->Bd, v, uout);
  VecScale(uout, -1.0);
  MatMultAdd(*shellctx->Ad, u, uout, uout);
  // Constant part vout = Adv + Bdu
  MatMult(*shellctx->Ad, v, vout);
  MatMultAdd(*shellctx->Bd, u, vout, vout);


  /* Control terms and Jaynes-Cummings coupling terms */
  int id_kl = 0; // index for accessing Ad_kl inside Ad_vec
  for (int iosc = 0; iosc < shellctx->nlevels.size(); iosc++) {

    /* Get controls */
    double p = shellctx->control_Re[iosc];
    double q = shellctx->control_Im[iosc];

    // uout += q^k*Acu
    MatMult((*(shellctx->Ac_vec))[iosc], u, *shellctx->aux);
    VecAXPY(uout, q, *shellctx->aux);
    // uout -= p^kBcv
    MatMult((*(shellctx->Bc_vec))[iosc], v, *shellctx->aux);
    VecAXPY(uout, -1.*p, *shellctx->aux);
    // vout += q^kAcv
    MatMult((*(shellctx->Ac_vec))[iosc], v, *shellctx->aux);
    VecAXPY(vout, q, *shellctx->aux);
    // vout += p^kBcu
    MatMult((*(shellctx->Bc_vec))[iosc], u, *shellctx->aux);
    VecAXPY(vout, p, *shellctx->aux);

    // Coupling terms
    for (int josc=iosc+1; josc<shellctx->nlevels.size(); josc++){

      double Jkl = shellctx->Jkl[id_kl]; 
      if (fabs(Jkl) > 1e-12) {

        double etakl = shellctx->eta[id_kl];
        double coskl = cos(etakl * shellctx->time);
        double sinkl = sin(etakl * shellctx->time);
        // uout += J_kl*sin*Adklu
        MatMult((*(shellctx->Ad_vec))[id_kl], u, *shellctx->aux);
        VecAXPY(uout, Jkl*sinkl, *shellctx->aux);
        // uout += -Jkl*cos*Bdklv
        MatMult((*(shellctx->Bd_vec))[id_kl], v, *shellctx->aux);
        VecAXPY(uout, -Jkl*coskl, *shellctx->aux);
        // vout += Jkl*cos*Bdklu
        MatMult((*(shellctx->Bd_vec))[id_kl], u, *shellctx->aux);
        VecAXPY(vout, Jkl*coskl, *shellctx->aux);
        //vout += Jkl*sin*Adklv
        MatMult((*(shellctx->Ad_vec))[id_kl], v, *shellctx->aux);
        VecAXPY(vout, Jkl*sinkl, *shellctx->aux);
      }
      id_kl++;
    }
  }

  /* Restore */
  VecRestoreSubVector(x, *shellctx->isu, &u);
  VecRestoreSubVector(x, *shellctx->isv, &v);
  VecRestoreSubVector(y, *shellctx->isu, &uout);
  VecRestoreSubVector(y, *shellctx->isv, &vout);

  return 0;
}


/* Sparse-matrix solver: Define the action of RHS^T on a vector x */
int myMatMultTranspose_sparsemat(Mat RHS, Vec x, Vec y) {

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
          // + sum_kl J_kl*sin(eta_kl*t) * Ad_kl^T * u
          //         +J_kl*cos(eta_kl*t) * Bd_kl^T * v  ]   cross terms
  // vout = -Im^T*u + Re^T*v
  //      = -(Bd + sum_k p_kB_k)^T*u + (Ad + sum_k q_kA_k)^T*v
        // + sum_kl - J_kl*cos(eta_kl*t) * Bd_kl^T * u
        //          + J_kl*sin(eta_kl*t) * Ad_kl^T * v  ]   cross terms

  // Constant part uout = Ad^Tu + Bd^Tv
  MatMultTranspose(*shellctx->Bd, v, uout);
  MatMultTransposeAdd(*shellctx->Ad, u, uout, uout);
  // Constant part vout = -Bd^Tu + Ad^Tv
  MatMultTranspose(*shellctx->Bd, u, vout);
  VecScale(vout, -1.0);
  MatMultTransposeAdd(*shellctx->Ad, v, vout, vout);

  /* Control and coupling term */
  int id_kl = 0; // index for accessing Ad_kl inside Ad_vec
  for (int iosc = 0; iosc < shellctx->nlevels.size(); iosc++) {
    /* Get controls */
    double p = shellctx->control_Re[iosc];
    double q = shellctx->control_Im[iosc];

    // uout += q^k*Ac^Tu
    MatMultTranspose((*(shellctx->Ac_vec))[iosc], u, *shellctx->aux);
    VecAXPY(uout, q, *shellctx->aux);
    // uout += p^kBc^Tv
    MatMultTranspose((*(shellctx->Bc_vec))[iosc], v, *shellctx->aux);
    VecAXPY(uout, p, *shellctx->aux);
    // vout += q^kAc^Tv
    MatMultTranspose((*(shellctx->Ac_vec))[iosc], v, *shellctx->aux);
    VecAXPY(vout, q, *shellctx->aux);
    // vout -= p^kBc^Tu
    MatMultTranspose((*(shellctx->Bc_vec))[iosc], u, *shellctx->aux);
    VecAXPY(vout, -1.*p, *shellctx->aux);

    // Coupling terms
    for (int josc=iosc+1; josc<shellctx->nlevels.size(); josc++){
      double Jkl = shellctx->Jkl[id_kl]; 

      if (fabs(Jkl) > 1e-12) {
        double etakl = shellctx->eta[id_kl];
        double coskl = cos(etakl * shellctx->time);
        double sinkl = sin(etakl * shellctx->time);
        // uout += J_kl*sin*Adklu^T
        MatMultTranspose((*(shellctx->Ad_vec))[id_kl], u, *shellctx->aux);
        VecAXPY(uout, Jkl*sinkl, *shellctx->aux);
        // uout += +Jkl*cos*Bdklv^T
        MatMultTranspose((*(shellctx->Bd_vec))[id_kl], v, *shellctx->aux);
        VecAXPY(uout,  Jkl*coskl, *shellctx->aux);
        // vout += - Jkl*cos*Bdklu^T
        MatMultTranspose((*(shellctx->Bd_vec))[id_kl], u, *shellctx->aux);
        VecAXPY(vout, - Jkl*coskl, *shellctx->aux);
        //vout += Jkl*sin*Adklv^T
        MatMultTranspose((*(shellctx->Ad_vec))[id_kl], v, *shellctx->aux);
        VecAXPY(vout, Jkl*sinkl, *shellctx->aux);
      }
      id_kl++;
    }
  }

  /* Restore */
  VecRestoreSubVector(x, *shellctx->isu, &u);
  VecRestoreSubVector(x, *shellctx->isv, &v);
  VecRestoreSubVector(y, *shellctx->isu, &uout);
  VecRestoreSubVector(y, *shellctx->isv, &vout);

  return 0;
}



/* Matfree-solver for 2 Oscillators: Define the action of RHS on a vector x */
template <int n0, int n1>
int myMatMult_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);


  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)
    decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2)
    dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)
    decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2)
    dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double cos01 = cos(eta01 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
  int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
  int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {

          /* --- Diagonal part ---*/
          //Get input x values
          double xre = xptr[2 * it];
          double xim = xptr[2 * it + 1];
          // drift Hamiltonian: uout = ( hd(ik) - hd(ik'))*vin
          //                    vout = (-hd(ik) + hd(ik'))*uin
          double hd  = H_detune(detuning_freq0, detuning_freq1, i0, i1)
                     + H_selfkerr(xi0, xi1, i0, i1)
                     + H_crosskerr(xi01, i0, i1);
          double hdp = H_detune(detuning_freq0, detuning_freq1, i0p, i1p)
                     + H_selfkerr(xi0, xi1, i0p, i1p)
                     + H_crosskerr(xi01, i0p, i1p);
          double yre = ( hd - hdp ) * xim;
          double yim = (-hd + hdp ) * xre;
          // Decay l1, diagonal part: xout += l1diag xin
          // Dephasing l2: xout += l2(ik, ikp) xin
          double l1diag = L1diag(decay0, decay1, i0, i1, i0p, i1p);
          double l2 = L2(dephase0, dephase1, i0, i1, i0p, i1p);
          yre += (l2 + l1diag) * xre;
          yim += (l2 + l1diag) * xim;


          /* --- Offdiagonal: Jkl coupling term --- */
          // oscillator 0<->1 
          Jkl_coupling(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);

          /* --- Offdiagonal part of decay L1 */
          // Oscillators 0
          L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
          // Oscillator 1
          L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);

          /* --- Control hamiltonian --- */
          // Oscillator 0 
          control(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
          // Oscillator 1
          control(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);

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


/* Matrix-free solver for 2 Oscillators: Define the action of RHS^T on a vector x */
template <int n0, int n1>
int myMatMultTranspose_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);

  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();
  double xi01 = shellctx->crosskerr[0];  // zz-coupling 
  double J01 = shellctx->Jkl[0];   // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)
    decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2)
    dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)
    decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2)
    dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double cos01 = cos(eta01 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
  int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
  int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);


  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {
        for (int i1 = 0; i1 < n1; i1++)  {

          /* --- Diagonal part ---*/
          //Get input x values
          double xre = xptr[2 * it];
          double xim = xptr[2 * it + 1];
          // drift Hamiltonian Hd^T: uout = ( hd(ik) - hd(ik'))*vin
          //                         vout = (-hd(ik) + hd(ik'))*uin
          double hd  = H_detune(detuning_freq0, detuning_freq1, i0, i1)
                     + H_selfkerr(xi0, xi1, i0, i1)
                     + H_crosskerr(xi01, i0, i1);
          double hdp = H_detune(detuning_freq0, detuning_freq1, i0p, i1p)
                     + H_selfkerr(xi0, xi1, i0p, i1p)
                     + H_crosskerr(xi01, i0p, i1p);
          double yre = (-hd + hdp ) * xim;
          double yim = ( hd - hdp ) * xre;
          // Decay l1^T, diagonal part: xout += l1diag xin
          // Dephasing l2^T: xout += l2(ik, ikp) xin
          double l1diag = L1diag(decay0, decay1, i0, i1, i0p, i1p);
          double l2 = L2(dephase0, dephase1, i0, i1, i0p, i1p);
          yre += (l2 + l1diag) * xre;
          yim += (l2 + l1diag) * xim;

          /* --- Offdiagonal coupling term J_kl --- */
          // oscillator 0<->1
          Jkl_coupling_T(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
 
          /* --- Offdiagonal part of decay L1^T */
          // Oscillators 0
          L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
          // Oscillator 1
          L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);

          /* --- Control hamiltonian  --- */
          // Oscillator 0
          control_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
          // Oscillator 1
          control_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);


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


/* Matfree-solver for 3 Oscillators: Define the action of RHS on a vector x */
template <int n0, int n1, int n2>
int myMatMult_matfree(Mat RHS, Vec x, Vec y){

   /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr); 
  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi2  = shellctx->oscil_vec[2]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double xi02 = shellctx->crosskerr[1];  // zz-coupling
  double xi12 = shellctx->crosskerr[2];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double J02  = shellctx->Jkl[1];  // Jaynes-Cummings coupling
  double J12  = shellctx->Jkl[2];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double eta02 = shellctx->eta[1];
  double eta12 = shellctx->eta[2];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double detuning_freq2 = shellctx->oscil_vec[2]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double decay2 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  double dephase2= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)   decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)   decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  if (shellctx->oscil_vec[2]->getDecayTime() > 1e-14 && shellctx->addT1)   decay2= 1./shellctx->oscil_vec[2]->getDecayTime();
  if (shellctx->oscil_vec[2]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase2 = 1./shellctx->oscil_vec[2]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double pt2 = shellctx->control_Re[2];
  double qt2 = shellctx->control_Im[2];
  double cos01 = cos(eta01 * shellctx->time);
  double cos02 = cos(eta02 * shellctx->time);
  double cos12 = cos(eta12 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);
  double sin02 = sin(eta02 * shellctx->time);
  double sin12 = sin(eta12 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1,n2, 1,0,0,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1,n2, 0,1,0,0,0,0);
  int stridei2  = TensorGetIndex(n0,n1,n2, 0,0,1,0,0,0);
  int stridei0p = TensorGetIndex(n0,n1,n2, 0,0,0,1,0,0);
  int stridei1p = TensorGetIndex(n0,n1,n2, 0,0,0,0,1,0);
  int stridei2p = TensorGetIndex(n0,n1,n2, 0,0,0,0,0,1);

   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i2p = 0; i2p < n2; i2p++)  {
        for (int i0 = 0; i0 < n0; i0++)  {
          for (int i1 = 0; i1 < n1; i1++)  {
            for (int i2 = 0; i2 < n2; i2++)  {

              /* --- Diagonal part ---*/
              //Get input x values
              double xre = xptr[2 * it];
              double xim = xptr[2 * it + 1];
              // drift Hamiltonian: uout = ( hd(ik) - hd(ik'))*vin
              //                    vout = (-hd(ik) + hd(ik'))*uin
              double hd  = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, i0, i1, i2)
                         + H_selfkerr(xi0, xi1, xi2, i0, i1, i2)
                         + H_crosskerr(xi01, xi02, xi12, i0, i1, i2);
              double hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, i0p, i1p, i2p)
                         + H_selfkerr(xi0, xi1, xi2, i0p, i1p, i2p)
                         + H_crosskerr(xi01, xi02, xi12, i0p, i1p, i2p);
              double yre = ( hd - hdp ) * xim;
              double yim = (-hd + hdp ) * xre;
              // Decay l1, diagonal part: xout += l1diag xin
              // Dephasing l2: xout += l2(ik, ikp) xin
              double l1diag = L1diag(decay0, decay1, decay2, i0, i1, i2, i0p, i1p, i2p);
              double l2 = L2(dephase0, dephase1, dephase2, i0, i1, i2, i0p, i1p, i2p);
              yre += (l2 + l1diag) * xre;
              yim += (l2 + l1diag) * xim;


              /* --- Offdiagonal: Jkl coupling  --- */
              // oscillator 0<->1 
              Jkl_coupling(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
              // oscillator 0<->2
              Jkl_coupling(it, n0, n2, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
              // oscillator 1<->2
              Jkl_coupling(it, n1, n2, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);

              /* --- Offdiagonal part of decay L1 */
              // Oscillators 0
              L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
              // Oscillator 1
              L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
              // Oscillator 2
              L1decay(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
              

              /* --- Control hamiltonian ---  */
              // Oscillator 0 
              control(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
              // Oscillator 1
              control(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
              // Oscillator 1
              control(it, n2, i2, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
              
              /* --- Update --- */
              yptr[2*it]   = yre;
              yptr[2*it+1] = yim;
              it++;
            }
          }
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  return 0;
}

/* Matfree-solver for 3 Oscillators: Define the action of RHS^T on a vector x */
template <int n0, int n1, int n2>
int myMatMultTranspose_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);


  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi2  = shellctx->oscil_vec[2]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double xi02 = shellctx->crosskerr[1];  // zz-coupling
  double xi12 = shellctx->crosskerr[2];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double J02  = shellctx->Jkl[1];  // Jaynes-Cummings coupling
  double J12  = shellctx->Jkl[2];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double eta02 = shellctx->eta[1];
  double eta12 = shellctx->eta[2];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double detuning_freq2 = shellctx->oscil_vec[2]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double decay2 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  double dephase2= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)   decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)   decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  if (shellctx->oscil_vec[2]->getDecayTime() > 1e-14 && shellctx->addT1)   decay2= 1./shellctx->oscil_vec[2]->getDecayTime();
  if (shellctx->oscil_vec[2]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase2 = 1./shellctx->oscil_vec[2]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double pt2 = shellctx->control_Re[2];
  double qt2 = shellctx->control_Im[2];
  double cos01 = cos(eta01 * shellctx->time);
  double cos02 = cos(eta02 * shellctx->time);
  double cos12 = cos(eta12 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);
  double sin02 = sin(eta02 * shellctx->time);
  double sin12 = sin(eta12 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1,n2, 1,0,0,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1,n2, 0,1,0,0,0,0);
  int stridei2  = TensorGetIndex(n0,n1,n2, 0,0,1,0,0,0);
  int stridei0p = TensorGetIndex(n0,n1,n2, 0,0,0,1,0,0);
  int stridei1p = TensorGetIndex(n0,n1,n2, 0,0,0,0,1,0);
  int stridei2p = TensorGetIndex(n0,n1,n2, 0,0,0,0,0,1);

   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i2p = 0; i2p < n2; i2p++)  {
        for (int i0 = 0; i0 < n0; i0++)  {
          for (int i1 = 0; i1 < n1; i1++)  {
            for (int i2 = 0; i2 < n2; i2++)  {

              /* --- Diagonal part ---*/
              //Get input x values
              double xre = xptr[2 * it];
              double xim = xptr[2 * it + 1];
              // drift Hamiltonian Hd^T: uout = ( hd(ik) - hd(ik'))*vin
              //                         vout = (-hd(ik) + hd(ik'))*uin
              double hd  = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, i0, i1, i2)
                         + H_selfkerr(xi0, xi1, xi2, i0, i1, i2)
                         + H_crosskerr(xi01, xi02, xi12, i0, i1, i2);
              double hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, i0p, i1p, i2p)
                         + H_selfkerr(xi0, xi1, xi2, i0p, i1p, i2p)
                         + H_crosskerr(xi01, xi02, xi12, i0p, i1p, i2p);
              double yre = (-hd + hdp ) * xim;
              double yim = ( hd - hdp ) * xre;
              // Decay l1^T, diagonal part: xout += l1diag xin
              // Dephasing l2^T: xout += l2(ik, ikp) xin
              double l1diag = L1diag(decay0, decay1, decay2, i0, i1, i2, i0p, i1p, i2p);
              double l2 = L2(dephase0, dephase1, dephase2, i0, i1, i2, i0p, i1p, i2p);
              yre += (l2 + l1diag) * xre;
              yim += (l2 + l1diag) * xim;

              /* --- Offdiagonal coupling term J_kl --- */
              // oscillator 0<->1
              Jkl_coupling_T(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
              // oscillator 0<->2
              Jkl_coupling_T(it, n0, n2, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
              // oscillator 1<->2
              Jkl_coupling_T(it, n1, n2, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
              

              /* --- Offdiagonal part of decay L1^T */
              // Oscillators 0
              L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
              // Oscillator 1
              L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
             // Oscillator 2
              L1decay_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);

              /* --- Control hamiltonian  --- */
              // Oscillator 0
              control_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
              // Oscillator 1
              control_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
              // Oscillator 2
              control_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);

              /* Update */
              yptr[2*it]   = yre;
              yptr[2*it+1] = yim;
              it++;
            }
          }
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  return 0;
}



/* Matfree-solver for 4 Oscillators: Define the action of RHS on a vector x */
template <int n0, int n1, int n2, int n3>
int myMatMult_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr); 

  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi2  = shellctx->oscil_vec[2]->getSelfkerr();   
  double xi3  = shellctx->oscil_vec[3]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double xi02 = shellctx->crosskerr[1];  // zz-coupling
  double xi03 = shellctx->crosskerr[2];  // zz-coupling
  double xi12 = shellctx->crosskerr[3];  // zz-coupling
  double xi13 = shellctx->crosskerr[4];  // zz-coupling
  double xi23 = shellctx->crosskerr[5];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double J02  = shellctx->Jkl[1];  // Jaynes-Cummings coupling
  double J03  = shellctx->Jkl[2];  // Jaynes-Cummings coupling
  double J12  = shellctx->Jkl[3];  // Jaynes-Cummings coupling
  double J13  = shellctx->Jkl[4];  // Jaynes-Cummings coupling
  double J23  = shellctx->Jkl[5];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double eta02 = shellctx->eta[1];
  double eta03 = shellctx->eta[2];
  double eta12 = shellctx->eta[3];
  double eta13 = shellctx->eta[4];
  double eta23 = shellctx->eta[5];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double detuning_freq2 = shellctx->oscil_vec[2]->getDetuning();
  double detuning_freq3 = shellctx->oscil_vec[3]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double decay2 = 0.0;
  double decay3 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  double dephase2= 0.0;
  double dephase3= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)   decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)   decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  if (shellctx->oscil_vec[2]->getDecayTime() > 1e-14 && shellctx->addT1)   decay2= 1./shellctx->oscil_vec[2]->getDecayTime();
  if (shellctx->oscil_vec[2]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase2 = 1./shellctx->oscil_vec[2]->getDephaseTime();
  if (shellctx->oscil_vec[3]->getDecayTime() > 1e-14 && shellctx->addT1)   decay3= 1./shellctx->oscil_vec[3]->getDecayTime();
  if (shellctx->oscil_vec[3]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase3 = 1./shellctx->oscil_vec[3]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double pt2 = shellctx->control_Re[2];
  double qt2 = shellctx->control_Im[2];
  double pt3 = shellctx->control_Re[3];
  double qt3 = shellctx->control_Im[3];
  double cos01 = cos(eta01 * shellctx->time);
  double cos02 = cos(eta02 * shellctx->time);
  double cos03 = cos(eta03 * shellctx->time);
  double cos12 = cos(eta12 * shellctx->time);
  double cos13 = cos(eta13 * shellctx->time);
  double cos23 = cos(eta23 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);
  double sin02 = sin(eta02 * shellctx->time);
  double sin03 = sin(eta03 * shellctx->time);
  double sin12 = sin(eta12 * shellctx->time);
  double sin13 = sin(eta13 * shellctx->time);
  double sin23 = sin(eta23 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1,n2,n3, 1,0,0,0,0,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1,n2,n3, 0,1,0,0,0,0,0,0);
  int stridei2  = TensorGetIndex(n0,n1,n2,n3, 0,0,1,0,0,0,0,0);
  int stridei3  = TensorGetIndex(n0,n1,n2,n3, 0,0,0,1,0,0,0,0);
  int stridei0p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,1,0,0,0);
  int stridei1p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,1,0,0);
  int stridei2p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,0,1,0);
  int stridei3p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,0,0,1);

   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i2p = 0; i2p < n2; i2p++)  {
        for (int i3p = 0; i3p < n3; i3p++)  {
          for (int i0 = 0; i0 < n0; i0++)  {
            for (int i1 = 0; i1 < n1; i1++)  {
              for (int i2 = 0; i2 < n2; i2++)  {
                for (int i3 = 0; i3 < n3; i3++)  {

                  /* --- Diagonal part ---*/
                  double xre = xptr[2 * it];
                  double xim = xptr[2 * it + 1];
                  // drift Hamiltonian: uout = ( hd(ik) - hd(ik'))*vin
                  //                    vout = (-hd(ik) + hd(ik'))*uin
                  double hd  = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, i0, i1, i2, i3)
                             + H_selfkerr(xi0, xi1, xi2, xi3, i0, i1, i2, i3)
                             + H_crosskerr(xi01, xi02, xi03, xi12, xi13, xi23, i0, i1, i2, i3);
                  double hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, i0p, i1p, i2p, i3p)
                             + H_selfkerr(xi0, xi1, xi2, xi3, i0p, i1p, i2p, i3p)
                             + H_crosskerr(xi01, xi02, xi03, xi12, xi13, xi23, i0p, i1p, i2p, i3p);
                  double yre = ( hd - hdp ) * xim;
                  double yim = (-hd + hdp ) * xre;
                  // Decay l1, diagonal part: xout += l1diag xin
                  // Dephasing l2: xout += l2(ik, ikp) xin
                  double l1diag = L1diag(decay0, decay1, decay2, decay3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                  double l2 = L2(dephase0, dephase1, dephase2, dephase3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                  yre += (l2 + l1diag) * xre;
                  yim += (l2 + l1diag) * xim;


                  /* --- Offdiagonal: Jkl coupling  --- */
                  // oscillator 0<->1 
                  Jkl_coupling(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                  // oscillator 0<->2
                  Jkl_coupling(it, n0, n2, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                  // oscillator 0<->3
                  Jkl_coupling(it, n0, n3, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                  // oscillator 1<->2
                  Jkl_coupling(it, n1, n2, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                  // oscillator 1<->3
                  Jkl_coupling(it, n1, n3, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                  // oscillator 2<->3
                  Jkl_coupling(it, n2, n3, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);

                  /* --- Offdiagonal part of decay L1 */
                  // Oscillators 0
                  L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                  // Oscillator 1
                  L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                  // Oscillator 2
                  L1decay(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
                  // Oscillator 3
                  L1decay(it, n3, i3, i3p, stridei3, stridei3p, xptr, decay3, &yre, &yim);
              

                  /* --- Control hamiltonian ---  */
                  // Oscillator 0 
                  control(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                  // Oscillator 1
                  control(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                  // Oscillator 2
                  control(it, n2, i2, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                  // Oscillator 2
                  control(it, n3, i3, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);
              
                  /* --- Update --- */
                  yptr[2*it]   = yre;
                  yptr[2*it+1] = yim;
                  it++;
                }
              }
            }
          }
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  return 0;
}


/* Matfree-solver for 4 Oscillators: Define the action of RHS^T on a vector x */
template <int n0, int n1, int n2, int n3>
int myMatMultTranspose_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);

  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi2  = shellctx->oscil_vec[2]->getSelfkerr();   
  double xi3  = shellctx->oscil_vec[3]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double xi02 = shellctx->crosskerr[1];  // zz-coupling
  double xi03 = shellctx->crosskerr[2];  // zz-coupling
  double xi12 = shellctx->crosskerr[3];  // zz-coupling
  double xi13 = shellctx->crosskerr[4];  // zz-coupling
  double xi23 = shellctx->crosskerr[5];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double J02  = shellctx->Jkl[1];  // Jaynes-Cummings coupling
  double J03  = shellctx->Jkl[2];  // Jaynes-Cummings coupling
  double J12  = shellctx->Jkl[3];  // Jaynes-Cummings coupling
  double J13  = shellctx->Jkl[4];  // Jaynes-Cummings coupling
  double J23  = shellctx->Jkl[5];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double eta02 = shellctx->eta[1];
  double eta03 = shellctx->eta[2];
  double eta12 = shellctx->eta[3];
  double eta13 = shellctx->eta[4];
  double eta23 = shellctx->eta[5];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double detuning_freq2 = shellctx->oscil_vec[2]->getDetuning();
  double detuning_freq3 = shellctx->oscil_vec[3]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double decay2 = 0.0;
  double decay3 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  double dephase2= 0.0;
  double dephase3= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)   decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)   decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  if (shellctx->oscil_vec[2]->getDecayTime() > 1e-14 && shellctx->addT1)   decay2= 1./shellctx->oscil_vec[2]->getDecayTime();
  if (shellctx->oscil_vec[2]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase2 = 1./shellctx->oscil_vec[2]->getDephaseTime();
  if (shellctx->oscil_vec[3]->getDecayTime() > 1e-14 && shellctx->addT1)   decay3= 1./shellctx->oscil_vec[3]->getDecayTime();
  if (shellctx->oscil_vec[3]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase3 = 1./shellctx->oscil_vec[3]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double pt2 = shellctx->control_Re[2];
  double qt2 = shellctx->control_Im[2];
  double pt3 = shellctx->control_Re[3];
  double qt3 = shellctx->control_Im[3];
  double cos01 = cos(eta01 * shellctx->time);
  double cos02 = cos(eta02 * shellctx->time);
  double cos03 = cos(eta03 * shellctx->time);
  double cos12 = cos(eta12 * shellctx->time);
  double cos13 = cos(eta13 * shellctx->time);
  double cos23 = cos(eta23 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);
  double sin02 = sin(eta02 * shellctx->time);
  double sin03 = sin(eta03 * shellctx->time);
  double sin12 = sin(eta12 * shellctx->time);
  double sin13 = sin(eta13 * shellctx->time);
  double sin23 = sin(eta23 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1,n2,n3, 1,0,0,0,0,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1,n2,n3, 0,1,0,0,0,0,0,0);
  int stridei2  = TensorGetIndex(n0,n1,n2,n3, 0,0,1,0,0,0,0,0);
  int stridei3  = TensorGetIndex(n0,n1,n2,n3, 0,0,0,1,0,0,0,0);
  int stridei0p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,1,0,0,0);
  int stridei1p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,1,0,0);
  int stridei2p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,0,1,0);
  int stridei3p = TensorGetIndex(n0,n1,n2,n3, 0,0,0,0,0,0,0,1);


   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i2p = 0; i2p < n2; i2p++)  {
        for (int i3p = 0; i3p < n3; i3p++)  {
          for (int i0 = 0; i0 < n0; i0++)  {
            for (int i1 = 0; i1 < n1; i1++)  {
              for (int i2 = 0; i2 < n2; i2++)  {
                for (int i3 = 0; i3 < n3; i3++)  {
                  double xre = xptr[2 * it];
                  double xim = xptr[2 * it + 1];

                  /* --- Diagonal part ---*/
                  // drift Hamiltonian Hd^T: uout = ( hd(ik) - hd(ik'))*vin
                  //                         vout = (-hd(ik) + hd(ik'))*uin
                  double hd  = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, i0, i1, i2, i3)
                             + H_selfkerr(xi0, xi1, xi2, xi3, i0, i1, i2, i3)
                             + H_crosskerr(xi01, xi02, xi03, xi12, xi13, xi23, i0, i1, i2, i3);
                  double hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, i0p, i1p, i2p, i3p)
                             + H_selfkerr(xi0, xi1, xi2, xi3, i0p, i1p, i2p, i3p)
                             + H_crosskerr(xi01, xi02, xi03, xi12, xi13, xi23, i0p, i1p, i2p, i3p);
                  double yre = (-hd + hdp ) * xim;
                  double yim = ( hd - hdp ) * xre;
                  // Decay l1^T, diagonal part: xout += l1diag xin
                  // Dephasing l2^T: xout += l2(ik, ikp) xin
                  double l1diag = L1diag(decay0, decay1, decay2, decay3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                  double l2 = L2(dephase0, dephase1, dephase2, dephase3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                  yre += (l2 + l1diag) * xre;
                  yim += (l2 + l1diag) * xim;

                  /* --- Offdiagonal coupling term J_kl --- */
                  // oscillator 0<->1
                  Jkl_coupling_T(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                  // oscillator 0<->2
                  Jkl_coupling_T(it, n0, n2, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                  // oscillator 0<->3
                  Jkl_coupling_T(it, n0, n3, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                  // oscillator 1<->2
                  Jkl_coupling_T(it, n1, n2, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                  // oscillator 1<->3
                  Jkl_coupling_T(it, n1, n3, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                  // oscillator 2<->3
                  Jkl_coupling_T(it, n2, n3, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);
              

                  /* --- Offdiagonal part of decay L1^T */
                  // Oscillators 0
                  L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                  // Oscillator 1
                  L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                  // Oscillator 2
                  L1decay_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
                  // Oscillator 3
                  L1decay_T(it, n3, i3, i3p, stridei3, stridei3p, xptr, decay3, &yre, &yim);

                  /* --- Control hamiltonian  --- */
                  // Oscillator 0
                  control_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                  // Oscillator 1
                  control_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                  // Oscillator 2
                  control_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                  // Oscillator 3
                  control_T(it, n3, i3, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);

                  /* Update */
                  yptr[2*it]   = yre;
                  yptr[2*it+1] = yim;
                  it++;
                }
              }
            }
          }
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  return 0;
}


/* Matfree-solver for 5 Oscillators: Define the action of RHS on a vector x */
template <int n0, int n1, int n2, int n3, int n4>
int myMatMult_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr); 

  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi2  = shellctx->oscil_vec[2]->getSelfkerr();   
  double xi3  = shellctx->oscil_vec[3]->getSelfkerr();   
  double xi4  = shellctx->oscil_vec[4]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double xi02 = shellctx->crosskerr[1];  // zz-coupling
  double xi03 = shellctx->crosskerr[2];  // zz-coupling
  double xi04 = shellctx->crosskerr[3];  // zz-coupling
  double xi12 = shellctx->crosskerr[4];  // zz-coupling
  double xi13 = shellctx->crosskerr[5];  // zz-coupling
  double xi14 = shellctx->crosskerr[6];  // zz-coupling
  double xi23 = shellctx->crosskerr[7];  // zz-coupling
  double xi24 = shellctx->crosskerr[8];  // zz-coupling
  double xi34 = shellctx->crosskerr[9];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double J02  = shellctx->Jkl[1];  // Jaynes-Cummings coupling
  double J03  = shellctx->Jkl[2];  // Jaynes-Cummings coupling
  double J04  = shellctx->Jkl[3];  // Jaynes-Cummings coupling
  double J12  = shellctx->Jkl[4];  // Jaynes-Cummings coupling
  double J13  = shellctx->Jkl[5];  // Jaynes-Cummings coupling
  double J14  = shellctx->Jkl[6];  // Jaynes-Cummings coupling
  double J23  = shellctx->Jkl[7];  // Jaynes-Cummings coupling
  double J24  = shellctx->Jkl[8];  // Jaynes-Cummings coupling
  double J34  = shellctx->Jkl[9];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double eta02 = shellctx->eta[1];
  double eta03 = shellctx->eta[2];
  double eta04 = shellctx->eta[3];
  double eta12 = shellctx->eta[4];
  double eta13 = shellctx->eta[5];
  double eta14 = shellctx->eta[6];
  double eta23 = shellctx->eta[7];
  double eta24 = shellctx->eta[8];
  double eta34 = shellctx->eta[9];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double detuning_freq2 = shellctx->oscil_vec[2]->getDetuning();
  double detuning_freq3 = shellctx->oscil_vec[3]->getDetuning();
  double detuning_freq4 = shellctx->oscil_vec[4]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double decay2 = 0.0;
  double decay3 = 0.0;
  double decay4 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  double dephase2= 0.0;
  double dephase3= 0.0;
  double dephase4= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)   decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)   decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  if (shellctx->oscil_vec[2]->getDecayTime() > 1e-14 && shellctx->addT1)   decay2= 1./shellctx->oscil_vec[2]->getDecayTime();
  if (shellctx->oscil_vec[2]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase2 = 1./shellctx->oscil_vec[2]->getDephaseTime();
  if (shellctx->oscil_vec[3]->getDecayTime() > 1e-14 && shellctx->addT1)   decay3= 1./shellctx->oscil_vec[3]->getDecayTime();
  if (shellctx->oscil_vec[3]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase3 = 1./shellctx->oscil_vec[3]->getDephaseTime();
  if (shellctx->oscil_vec[4]->getDecayTime() > 1e-14 && shellctx->addT1)   decay4= 1./shellctx->oscil_vec[4]->getDecayTime();
  if (shellctx->oscil_vec[4]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase4 = 1./shellctx->oscil_vec[4]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double pt2 = shellctx->control_Re[2];
  double qt2 = shellctx->control_Im[2];
  double pt3 = shellctx->control_Re[3];
  double qt3 = shellctx->control_Im[3];
  double pt4 = shellctx->control_Re[4];
  double qt4 = shellctx->control_Im[4];
  double cos01 = cos(eta01 * shellctx->time);
  double cos02 = cos(eta02 * shellctx->time);
  double cos03 = cos(eta03 * shellctx->time);
  double cos04 = cos(eta04 * shellctx->time);
  double cos12 = cos(eta12 * shellctx->time);
  double cos13 = cos(eta13 * shellctx->time);
  double cos14 = cos(eta14 * shellctx->time);
  double cos23 = cos(eta23 * shellctx->time);
  double cos24 = cos(eta24 * shellctx->time);
  double cos34 = cos(eta34 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);
  double sin02 = sin(eta02 * shellctx->time);
  double sin03 = sin(eta03 * shellctx->time);
  double sin04 = sin(eta04 * shellctx->time);
  double sin12 = sin(eta12 * shellctx->time);
  double sin13 = sin(eta13 * shellctx->time);
  double sin14 = sin(eta14 * shellctx->time);
  double sin23 = sin(eta23 * shellctx->time);
  double sin24 = sin(eta24 * shellctx->time);
  double sin34 = sin(eta34 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1,n2,n3,n4, 1,0,0,0,0,0,0,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1,n2,n3,n4, 0,1,0,0,0,0,0,0,0,0);
  int stridei2  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,1,0,0,0,0,0,0,0);
  int stridei3  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,1,0,0,0,0,0,0);
  int stridei4  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,1,0,0,0,0,0);
  int stridei0p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,1,0,0,0,0);
  int stridei1p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,1,0,0,0);
  int stridei2p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,1,0,0);
  int stridei3p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,0,1,0);
  int stridei4p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,0,0,1);

   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i2p = 0; i2p < n2; i2p++)  {
        for (int i3p = 0; i3p < n3; i3p++)  {
          for (int i4p = 0; i4p < n4; i4p++)  {
            for (int i0 = 0; i0 < n0; i0++)  {
              for (int i1 = 0; i1 < n1; i1++)  {
                for (int i2 = 0; i2 < n2; i2++)  {
                  for (int i3 = 0; i3 < n3; i3++)  {
                    for (int i4 = 0; i4 < n4; i4++)  {

                      /* --- Diagonal part ---*/
                      double xre = xptr[2 * it];
                      double xim = xptr[2 * it + 1];
                      // drift Hamiltonian: uout = ( hd(ik) - hd(ik'))*vin
                      //                    vout = (-hd(ik) + hd(ik'))*uin
                      double hd  = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, detuning_freq4, i0, i1, i2, i3, i4)
                                 + H_selfkerr(xi0, xi1, xi2, xi3, xi4, i0, i1, i2, i3, i4)
                                 + H_crosskerr(xi01, xi02, xi03, xi04, xi12, xi13, xi14, xi23, xi24, xi34, i0, i1, i2, i3, i4);
                      double hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, detuning_freq4, i0p, i1p, i2p, i3p, i4p)
                                 + H_selfkerr(xi0, xi1, xi2, xi3, xi4, i0p, i1p, i2p, i3p, i4p)
                                 + H_crosskerr(xi01, xi02, xi03, xi04, xi12, xi13, xi14, xi23, xi24, xi34, i0p, i1p, i2p, i3p, i4p);
                      double yre = ( hd - hdp ) * xim;
                      double yim = (-hd + hdp ) * xre;
                      // Decay l1, diagonal part: xout += l1diag xin
                      // Dephasing l2: xout += l2(ik, ikp) xin
                      double l1diag = L1diag(decay0, decay1, decay2, decay3, decay4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                      double l2 = L2(dephase0, dephase1, dephase2, dephase3, dephase4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                      yre += (l2 + l1diag) * xre;
                      yim += (l2 + l1diag) * xim;


                      /* --- Offdiagonal: Jkl coupling  --- */
                      // oscillator 0<->1 
                      Jkl_coupling(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                      // oscillator 0<->2
                      Jkl_coupling(it, n0, n2, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                      // oscillator 0<->3
                      Jkl_coupling(it, n0, n3, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                      // oscillator 0<->4
                      Jkl_coupling(it, n0, n4, i0, i0p, i4, i4p, stridei0, stridei0p, stridei4, stridei4p, xptr, J04, cos04, sin04, &yre, &yim);
                      // oscillator 1<->2
                      Jkl_coupling(it, n1, n2, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                      // oscillator 1<->3
                      Jkl_coupling(it, n1, n3, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                      // oscillator 1<->4
                      Jkl_coupling(it, n1, n4, i1, i1p, i4, i4p, stridei1, stridei1p, stridei4, stridei4p, xptr, J14, cos14, sin14, &yre, &yim);
                      // oscillator 2<->3
                      Jkl_coupling(it, n2, n3, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);
                      // oscillator 2<->4
                      Jkl_coupling(it, n2, n4, i2, i2p, i4, i4p, stridei2, stridei2p, stridei4, stridei4p, xptr, J24, cos24, sin24, &yre, &yim);
                      // oscillator 3<->4
                      Jkl_coupling(it, n3, n4, i3, i3p, i4, i4p, stridei3, stridei3p, stridei4, stridei4p, xptr, J34, cos34, sin34, &yre, &yim);

                      /* --- Offdiagonal part of decay L1 */
                      // Oscillator 0
                      L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                      // Oscillator 1
                      L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                      // Oscillator 2
                      L1decay(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
                      // Oscillator 3
                      L1decay(it, n3, i3, i3p, stridei3, stridei3p, xptr, decay3, &yre, &yim);
                      // Oscillator 4
                      L1decay(it, n4, i4, i4p, stridei4, stridei4p, xptr, decay4, &yre, &yim);
              

                      /* --- Control hamiltonian ---  */
                      // Oscillator 0 
                      control(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                      // Oscillator 1
                      control(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                      // Oscillator 2
                      control(it, n2, i2, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                      // Oscillator 3
                      control(it, n3, i3, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);
                      // Oscillator 4
                      control(it, n4, i4, i4p, stridei4, stridei4p, xptr, pt4, qt4, &yre, &yim);
              
                      /* --- Update --- */
                      yptr[2*it]   = yre;
                      yptr[2*it+1] = yim;
                      it++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  return 0;
}


/* Matfree-solver for 5 Oscillators: Define the action of RHS^T on a vector x */
template <int n0, int n1, int n2, int n3, int n4>
int myMatMultTranspose_matfree(Mat RHS, Vec x, Vec y){

  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  /* Get access to x and y */
  const double* xptr;
  double* yptr;
  VecGetArrayRead(x, &xptr);
  VecGetArray(y, &yptr);

  /* Evaluate coefficients */
  double xi0  = shellctx->oscil_vec[0]->getSelfkerr();
  double xi1  = shellctx->oscil_vec[1]->getSelfkerr();   
  double xi2  = shellctx->oscil_vec[2]->getSelfkerr();   
  double xi3  = shellctx->oscil_vec[3]->getSelfkerr();   
  double xi4  = shellctx->oscil_vec[4]->getSelfkerr();   
  double xi01 = shellctx->crosskerr[0];  // zz-coupling
  double xi02 = shellctx->crosskerr[1];  // zz-coupling
  double xi03 = shellctx->crosskerr[2];  // zz-coupling
  double xi04 = shellctx->crosskerr[3];  // zz-coupling
  double xi12 = shellctx->crosskerr[4];  // zz-coupling
  double xi13 = shellctx->crosskerr[5];  // zz-coupling
  double xi14 = shellctx->crosskerr[6];  // zz-coupling
  double xi23 = shellctx->crosskerr[7];  // zz-coupling
  double xi24 = shellctx->crosskerr[8];  // zz-coupling
  double xi34 = shellctx->crosskerr[9];  // zz-coupling
  double J01  = shellctx->Jkl[0];  // Jaynes-Cummings coupling
  double J02  = shellctx->Jkl[1];  // Jaynes-Cummings coupling
  double J03  = shellctx->Jkl[2];  // Jaynes-Cummings coupling
  double J04  = shellctx->Jkl[3];  // Jaynes-Cummings coupling
  double J12  = shellctx->Jkl[4];  // Jaynes-Cummings coupling
  double J13  = shellctx->Jkl[5];  // Jaynes-Cummings coupling
  double J14  = shellctx->Jkl[6];  // Jaynes-Cummings coupling
  double J23  = shellctx->Jkl[7];  // Jaynes-Cummings coupling
  double J24  = shellctx->Jkl[8];  // Jaynes-Cummings coupling
  double J34  = shellctx->Jkl[9];  // Jaynes-Cummings coupling
  double eta01 = shellctx->eta[0];
  double eta02 = shellctx->eta[1];
  double eta03 = shellctx->eta[2];
  double eta04 = shellctx->eta[3];
  double eta12 = shellctx->eta[4];
  double eta13 = shellctx->eta[5];
  double eta14 = shellctx->eta[6];
  double eta23 = shellctx->eta[7];
  double eta24 = shellctx->eta[8];
  double eta34 = shellctx->eta[9];
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double detuning_freq1 = shellctx->oscil_vec[1]->getDetuning();
  double detuning_freq2 = shellctx->oscil_vec[2]->getDetuning();
  double detuning_freq3 = shellctx->oscil_vec[3]->getDetuning();
  double detuning_freq4 = shellctx->oscil_vec[4]->getDetuning();
  double decay0 = 0.0;
  double decay1 = 0.0;
  double decay2 = 0.0;
  double decay3 = 0.0;
  double decay4 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  double dephase2= 0.0;
  double dephase3= 0.0;
  double dephase4= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)   decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  if (shellctx->oscil_vec[1]->getDecayTime() > 1e-14 && shellctx->addT1)   decay1= 1./shellctx->oscil_vec[1]->getDecayTime();
  if (shellctx->oscil_vec[1]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase1 = 1./shellctx->oscil_vec[1]->getDephaseTime();
  if (shellctx->oscil_vec[2]->getDecayTime() > 1e-14 && shellctx->addT1)   decay2= 1./shellctx->oscil_vec[2]->getDecayTime();
  if (shellctx->oscil_vec[2]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase2 = 1./shellctx->oscil_vec[2]->getDephaseTime();
  if (shellctx->oscil_vec[3]->getDecayTime() > 1e-14 && shellctx->addT1)   decay3= 1./shellctx->oscil_vec[3]->getDecayTime();
  if (shellctx->oscil_vec[3]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase3 = 1./shellctx->oscil_vec[3]->getDephaseTime();
  if (shellctx->oscil_vec[4]->getDecayTime() > 1e-14 && shellctx->addT1)   decay4= 1./shellctx->oscil_vec[4]->getDecayTime();
  if (shellctx->oscil_vec[4]->getDephaseTime() > 1e-14 && shellctx->addT2) dephase4 = 1./shellctx->oscil_vec[4]->getDephaseTime();
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double pt2 = shellctx->control_Re[2];
  double qt2 = shellctx->control_Im[2];
  double pt3 = shellctx->control_Re[3];
  double qt3 = shellctx->control_Im[3];
  double pt4 = shellctx->control_Re[4];
  double qt4 = shellctx->control_Im[4];
  double cos01 = cos(eta01 * shellctx->time);
  double cos02 = cos(eta02 * shellctx->time);
  double cos03 = cos(eta03 * shellctx->time);
  double cos04 = cos(eta04 * shellctx->time);
  double cos12 = cos(eta12 * shellctx->time);
  double cos13 = cos(eta13 * shellctx->time);
  double cos14 = cos(eta14 * shellctx->time);
  double cos23 = cos(eta23 * shellctx->time);
  double cos24 = cos(eta24 * shellctx->time);
  double cos34 = cos(eta34 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);
  double sin02 = sin(eta02 * shellctx->time);
  double sin03 = sin(eta03 * shellctx->time);
  double sin04 = sin(eta04 * shellctx->time);
  double sin12 = sin(eta12 * shellctx->time);
  double sin13 = sin(eta13 * shellctx->time);
  double sin14 = sin(eta14 * shellctx->time);
  double sin23 = sin(eta23 * shellctx->time);
  double sin24 = sin(eta24 * shellctx->time);
  double sin34 = sin(eta34 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1,n2,n3,n4, 1,0,0,0,0,0,0,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1,n2,n3,n4, 0,1,0,0,0,0,0,0,0,0);
  int stridei2  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,1,0,0,0,0,0,0,0);
  int stridei3  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,1,0,0,0,0,0,0);
  int stridei4  = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,1,0,0,0,0,0);
  int stridei0p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,1,0,0,0,0);
  int stridei1p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,1,0,0,0);
  int stridei2p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,1,0,0);
  int stridei3p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,0,1,0);
  int stridei4p = TensorGetIndex(n0,n1,n2,n3,n4, 0,0,0,0,0,0,0,0,0,1);

   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0; i0p++)  {
    for (int i1p = 0; i1p < n1; i1p++)  {
      for (int i2p = 0; i2p < n2; i2p++)  {
        for (int i3p = 0; i3p < n3; i3p++)  {
          for (int i4p = 0; i4p < n4; i4p++)  {
            for (int i0 = 0; i0 < n0; i0++)  {
              for (int i1 = 0; i1 < n1; i1++)  {
                for (int i2 = 0; i2 < n2; i2++)  {
                  for (int i3 = 0; i3 < n3; i3++)  {
                    for (int i4 = 0; i4 < n4; i4++)  {

                      double xre = xptr[2 * it];
                      double xim = xptr[2 * it + 1];

                      /* --- Diagonal part ---*/
                      // drift Hamiltonian Hd^T: uout = ( hd(ik) - hd(ik'))*vin
                      //                         vout = (-hd(ik) + hd(ik'))*uin
                      double hd  = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, detuning_freq4, i0, i1, i2, i3, i4)
                                 + H_selfkerr(xi0, xi1, xi2, xi3, xi4, i0, i1, i2, i3, i4)
                                 + H_crosskerr(xi01, xi02, xi03, xi04, xi12, xi13, xi14, xi23, xi24, xi34,i0, i1, i2, i3, i4);
                      double hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, detuning_freq4, i0p, i1p, i2p, i3p, i4p)
                                 + H_selfkerr(xi0, xi1, xi2, xi3, xi4, i0p, i1p, i2p, i3p, i4p)
                                 + H_crosskerr(xi01, xi02, xi03, xi04, xi12, xi13, xi14, xi23, xi24, xi34, i0p, i1p, i2p, i3p, i4p);
                      double yre = (-hd + hdp ) * xim;
                      double yim = ( hd - hdp ) * xre;
                      // Decay l1^T, diagonal part: xout += l1diag xin
                      // Dephasing l2^T: xout += l2(ik, ikp) xin
                      double l1diag = L1diag(decay0, decay1, decay2, decay3, decay4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                      double l2 = L2(dephase0, dephase1, dephase2, dephase3, dephase4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                      yre += (l2 + l1diag) * xre;
                      yim += (l2 + l1diag) * xim;

                      /* --- Offdiagonal coupling term J_kl --- */
                      // oscillator 0<->1
                      Jkl_coupling_T(it, n0, n1, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                      // oscillator 0<->2
                      Jkl_coupling_T(it, n0, n2, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                      // oscillator 0<->3
                      Jkl_coupling_T(it, n0, n3, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                      // oscillator 0<->4
                      Jkl_coupling_T(it, n0, n4, i0, i0p, i4, i4p, stridei0, stridei0p, stridei4, stridei4p, xptr, J04, cos04, sin04, &yre, &yim);
                      // oscillator 1<->2
                      Jkl_coupling_T(it, n1, n2, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                      // oscillator 1<->3
                      Jkl_coupling_T(it, n1, n3, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                      // oscillator 1<->4
                      Jkl_coupling_T(it, n1, n4, i1, i1p, i4, i4p, stridei1, stridei1p, stridei4, stridei4p, xptr, J14, cos14, sin14, &yre, &yim);
                      // oscillator 2<->3
                      Jkl_coupling_T(it, n2, n3, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);
                      // oscillator 2<->4
                      Jkl_coupling_T(it, n2, n4, i2, i2p, i4, i4p, stridei2, stridei2p, stridei4, stridei4p, xptr, J24, cos24, sin24, &yre, &yim);
                      // oscillator 3<->4
                      Jkl_coupling_T(it, n3, n4, i3, i3p, i4, i4p, stridei3, stridei3p, stridei4, stridei4p, xptr, J34, cos34, sin34, &yre, &yim);
              
                      /* --- Offdiagonal part of decay L1^T */
                      // Oscillators 0
                      L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                      // Oscillator 1
                      L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                      // Oscillator 2
                      L1decay_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
                      // Oscillator 3
                      L1decay_T(it, n3, i3, i3p, stridei3, stridei3p, xptr, decay3, &yre, &yim);
                      // Oscillator 4
                      L1decay_T(it, n4, i4, i4p, stridei4, stridei4p, xptr, decay4, &yre, &yim);

                      /* --- Control hamiltonian  --- */
                      // Oscillator 0
                      control_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                      // Oscillator 1
                      control_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                      // Oscillator 2
                      control_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                      // Oscillator 3
                      control_T(it, n3, i3, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);
                      // Oscillator 4
                      control_T(it, n4, i4, i4p, stridei4, stridei4p, xptr, pt4, qt4, &yre, &yim);

                      /* Update */
                      yptr[2*it]   = yre;
                      yptr[2*it+1] = yim;
                      it++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  return 0;
}

/* --- 2 Oscillator cases --- */
int myMatMult_matfree_2Osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  if      (n0==3 && n1==20)  return myMatMult_matfree<3,20>(RHS, x, y);
  else if (n0==3 && n1==10)  return myMatMult_matfree<3,10>(RHS, x, y);
  else if (n0==4 && n1==4)   return myMatMult_matfree<4,4>(RHS, x, y);
  else if (n0==1 && n1==1)   return myMatMult_matfree<1,1>(RHS, x, y);
  else if (n0==2 && n1==2)   return myMatMult_matfree<2,2>(RHS, x, y);
  else if (n0==3 && n1==3)   return myMatMult_matfree<3,3>(RHS, x, y);
  else if (n0==20 && n1==20) return myMatMult_matfree<20,20>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}
int myMatMultTranspose_matfree_2Osc(Mat RHS, Vec x, Vec y){
 /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  if      (n0==3 && n1==20)  return myMatMultTranspose_matfree<3,20>(RHS, x, y);
  else if (n0==3 && n1==10)  return myMatMultTranspose_matfree<3,10>(RHS, x, y);
  else if (n0==4 && n1==4)   return myMatMultTranspose_matfree<4,4>(RHS, x, y);
  else if (n0==1 && n1==1)   return myMatMultTranspose_matfree<1,1>(RHS, x, y);
  else if (n0==2 && n1==2)   return myMatMultTranspose_matfree<2,2>(RHS, x, y);
  else if (n0==3 && n1==3)   return myMatMultTranspose_matfree<3,3>(RHS, x, y);
  else if (n0==20 && n1==20) return myMatMultTranspose_matfree<20,20>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}


/* --- 3 Oscillator cases --- */
int myMatMult_matfree_3Osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  int n2 = shellctx->nlevels[2];
  if      (n0==2 && n1==2 && n2==2) return myMatMult_matfree<2,2,2>(RHS, x, y);
  else if (n0==2 && n1==3 && n2==4) return myMatMult_matfree<2,3,4>(RHS, x, y);
  else if (n0==3 && n1==3 && n2==3) return myMatMult_matfree<3,3,3>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}
int myMatMultTranspose_matfree_3Osc(Mat RHS, Vec x, Vec y){
 /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  int n2 = shellctx->nlevels[2];
  if      (n0==2 && n1==2 && n2==2)  return myMatMultTranspose_matfree<2,2,2>(RHS, x, y);
  else if (n0==2 && n1==3 && n2==4)  return myMatMultTranspose_matfree<2,3,4>(RHS, x, y);
  else if (n0==3 && n1==3 && n2==3)  return myMatMultTranspose_matfree<3,3,3>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}


/* --- 4 Oscillator cases --- */
int myMatMult_matfree_4Osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  int n2 = shellctx->nlevels[2];
  int n3 = shellctx->nlevels[3];
  if      (n0==2 && n1==2 && n2==2 && n3 == 2) return myMatMult_matfree<2,2,2,2>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}
int myMatMultTranspose_matfree_4Osc(Mat RHS, Vec x, Vec y){
 /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  int n2 = shellctx->nlevels[2];
  int n3 = shellctx->nlevels[3];
  if      (n0==2 && n1==2 && n2==2 && n3==2)  return myMatMultTranspose_matfree<2,2,2,2>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}


/* --- 5 Oscillator cases --- */
int myMatMult_matfree_5Osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  int n2 = shellctx->nlevels[2];
  int n3 = shellctx->nlevels[3];
  int n4 = shellctx->nlevels[4];
  if      (n0==2 && n1==2 && n2==2 && n3 == 2 && n4 == 2) return myMatMult_matfree<2,2,2,2,2>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}
int myMatMultTranspose_matfree_5Osc(Mat RHS, Vec x, Vec y){
 /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);

  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  int n2 = shellctx->nlevels[2];
  int n3 = shellctx->nlevels[3];
  int n4 = shellctx->nlevels[4];
  if      (n0==2 && n1==2 && n2==2 && n3==2 && n4==2)  return myMatMultTranspose_matfree<2,2,2,2,2>(RHS, x, y);
  else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}

