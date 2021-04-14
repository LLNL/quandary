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


MasterEq::MasterEq(std::vector<int> nlevels_, std::vector<int> nessential_, Oscillator** oscil_vec_, const std::vector<double> selfker_, const std::vector<double> crossker_, const std::vector<double> Jkl_, const std::vector<double> eta_, const std::vector<double> detuning_freq_, LindbladType lindbladtype, const std::vector<double> collapse_time_, bool usematfree_) {
  int ierr;

  nlevels = nlevels_;
  nessential = nessential_;
  noscillators = nlevels.size();
  oscil_vec = oscil_vec_;
  selfker = selfker_;
  crossker = crossker_;
  Jkl = Jkl_;
  eta = eta_;
  detuning_freq = detuning_freq_;
  collapse_time = collapse_time_;
  usematfree = usematfree_;

  int mpisize_petsc;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_petsc);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);

  /* Get dimensions */
  dim_rho = 1;
  dim_ess = 1;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    dim_rho *= oscil_vec[iosc]->getNLevels();
    dim_ess *= nessential[iosc];
  }
  dim = dim_rho*dim_rho; // density matrix: N \times N -> vectorized: N^2
  if (mpirank_petsc == 0) printf("System dimension (complex) N^2 = %d\n",dim);

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
    case NONE:
      break;
    case DECAY: 
      addT1 = true;
      addT2 = false;
      break;
    case DEPHASE:
      addT1 = false;
      addT2 = true;
      break;
    case BOTH:
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

  /* Allocate MatShell context for applying RHS */
  RHSctx.isu = &isu;
  RHSctx.isv = &isv;
  RHSctx.selfker = selfker;
  RHSctx.crossker = crossker;
  RHSctx.Jkl = Jkl;
  RHSctx.eta = eta;
  RHSctx.detuning_freq = detuning_freq;
  RHSctx.collapse_time = collapse_time;
  RHSctx.addT1 = addT1;
  RHSctx.addT2 = addT2;
  if (!usematfree){
    RHSctx.Ac_vec = &Ac_vec;
    RHSctx.Bc_vec = &Bc_vec;
    RHSctx.Ad_vec = &Ad_vec;
    RHSctx.Bd_vec = &Bd_vec;
    RHSctx.Ad = &Ad;
    RHSctx.Bd = &Bd;
    RHSctx.Acu = &Acu;
    RHSctx.Acv = &Acv;
    RHSctx.Bcu = &Bcu;
    RHSctx.Bcv = &Bcv;
    RHSctx.Adklu = &Adklu;
    RHSctx.Adklv = &Adklv;
    RHSctx.Bdklu = &Bdklu;
    RHSctx.Bdklv = &Bdklv;
  }
  RHSctx.nlevels = nlevels;
  RHSctx.oscil_vec = &oscil_vec;
  RHSctx.time = 0.0;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    RHSctx.control_Re.push_back(0.0);
    RHSctx.control_Im.push_back(0.0);
  }

  /* Set the MatMult routine for applying the RHS to a vector x */
  if (usematfree) {
    MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_matfree_2osc);
    MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_matfree_2Osc);
  }
  else {
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
        MatDestroy(&Ad_vec[i]);
        MatDestroy(&Bd_vec[i]);
      }
      VecDestroy(&Acu);
      VecDestroy(&Acv);
      VecDestroy(&Bcu);
      VecDestroy(&Bcv);
      VecDestroy(&Adklu);
      VecDestroy(&Adklv);
      VecDestroy(&Bdklu);
      VecDestroy(&Bdklv);
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
  int ilow, iupp;
  int r1,r2, r1a, r2a, r1b, r2b;
  int col, col1, col2;
  double val;
  // double val1, val2;

  /* Set up control Hamiltonian building blocks Ac, Bc */
  for (int iosc = 0; iosc < noscillators; iosc++) {

    /* Get dimensions */
    int nk     = oscil_vec[iosc]->nlevels;
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


    /* Compute dipole-dipole coupling building blocks */
    /* Ad_kl(t) =  I_N \kron (ak^Tal − akal^T) − (al^Tak − alak^T) \kron IN */
    /* Bd_kl(t) = -I_N \kron (ak^Tal + akal^T) + (al^Tak + alak_T) \kron IN */
    for (int josc=iosc+1; josc<noscillators; josc++){

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
      int nj     = oscil_vec[josc]->nlevels;
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

    int nk     = oscil_vec[iosc]->nlevels;
    int nprek  = oscil_vec[iosc]->dim_preOsc;
    int npostk = oscil_vec[iosc]->dim_postOsc;
    double xik = selfker[iosc] * 2. * M_PI;
    double detunek = detuning_freq[iosc] * 2. * M_PI;

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
      int nj     = oscil_vec[josc]->nlevels;
      int npostj = oscil_vec[josc]->dim_postOsc;
      double xikj = crossker[coupling_id] * 2. * M_PI;
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
      if (collapse_time[iosc*2]   > 1e-14) gammaT1 = 1./(collapse_time[iosc*2]);
      if (collapse_time[iosc*2+1] > 1e-14) gammaT2 = 1./(collapse_time[iosc*2+1]);

      // Dimensions 
      int nk     = oscil_vec[iosc]->nlevels;
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
  MatCreateVecs(Ac_vec[0], &Acu, NULL);
  MatCreateVecs(Ac_vec[0], &Acv, NULL);
  MatCreateVecs(Bc_vec[0], &Bcu, NULL);
  MatCreateVecs(Bc_vec[0], &Bcv, NULL);
  MatCreateVecs(Ad_vec[0], &Adklu, NULL);
  MatCreateVecs(Ad_vec[0], &Adklv, NULL);
  MatCreateVecs(Bd_vec[0], &Bdklu, NULL);
  MatCreateVecs(Bd_vec[0], &Bdklv, NULL);

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


  if (usematfree) { // matrix free solver
    const double* xptr, *xbarptr;
    VecGetArrayRead(x, &xptr);
    VecGetArrayRead(xbar, &xbarptr);

    /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
    int n0 = nlevels[0];
    int n1 = nlevels[1];
    int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
    int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
    int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
    int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);

    /* Collect coefficients for gradient  */
    double coeff_p_osc0 = 0.0;
    double coeff_q_osc0 = 0.0;
    double coeff_p_osc1 = 0.0;
    double coeff_q_osc1 = 0.0;
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
            double res_p_re = 0.0;
            double res_p_im = 0.0;
            double res_q_re = 0.0;
            double res_q_im = 0.0;
            /* ik+1..,ik'.. term */
            if (i0 < n0-1) {
              int itx = it + stridei0;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i0 + 1);
              res_p_re +=   sq * xim;
              res_p_im += - sq * xre;
              res_q_re +=   sq * xre;
              res_q_im +=   sq * xim;
            }
            /* \rho(ik..,ik'+1..) */
            if (i0p < n0-1) {
              int itx = it + stridei0p;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i0p + 1);
              res_p_re += - sq * xim;
              res_p_im += + sq * xre;
              res_q_re +=   sq * xre;
              res_q_im +=   sq * xim;
            }
            /* \rho(ik-1..,ik'..) */
            if (i0 > 0) {
              int itx = it - stridei0;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i0);
              res_p_re += + sq * xim;
              res_p_im += - sq * xre;
              res_q_re += - sq * xre;
              res_q_im += - sq * xim;
            }
            /* \rho(ik..,ik'-1..) */
            if (i0p > 0) {
              int itx = it - stridei0p;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i0p);
              res_p_re += - sq * xim;
              res_p_im += + sq * xre;
              res_q_re += - sq * xre;
              res_q_im += - sq * xim;
            }
            /* Update the coefficients */
            coeff_p_osc0 += res_p_re * xbarre + res_p_im * xbarim;
            coeff_q_osc0 += res_q_re * xbarre + res_q_im * xbarim;

            /* --- Oscillator 1 --- */
            res_p_re = 0.0;
            res_p_im = 0.0;
            res_q_re = 0.0;
            res_q_im = 0.0;
            /* ik+1..,ik'.. term */
            if (i1 < n1-1) {
              int itx = it + stridei1;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i1 + 1);
              res_p_re +=   sq * xim;
              res_p_im += - sq * xre;
              res_q_re +=   sq * xre;
              res_q_im +=   sq * xim;
            }
            /* \rho(ik..,ik'+1..) */
            if (i1p < n1-1) {
              int itx = it + stridei1p;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i1p + 1);
              res_p_re += - sq * xim;
              res_p_im += + sq * xre;
              res_q_re +=   sq * xre;
              res_q_im +=   sq * xim;
            }
            /* \rho(ik-1..,ik'..) */
            if (i1 > 0) {
              int itx = it - stridei1;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i1);
              res_p_re += + sq * xim;
              res_p_im += - sq * xre;
              res_q_re += - sq * xre;
              res_q_im += - sq * xim;
            }
            /* \rho(ik..,ik'-1..) */
            if (i1p > 0) {
              int itx = it - stridei1p;
              double xre = xptr[2 * itx];
              double xim = xptr[2 * itx + 1];
              double sq = sqrt(i1p);
              res_p_re += - sq * xim;
              res_p_im += + sq * xre;
              res_q_re += - sq * xre;
              res_q_im += - sq * xim;
            }
            coeff_p_osc1 += res_p_re * xbarre + res_p_im * xbarim;
            coeff_q_osc1 += res_q_re * xbarre + res_q_im * xbarim;
            it++;
          }
        }
      }
    }
    VecRestoreArrayRead(x, &xptr);
    VecRestoreArrayRead(xbar, &xbarptr);

    /* Set the gradient values */
    // Oscillator 0
    for (int i=0; i<nparams_max; i++){
      dRedp[i] = 0.0;
      dImdp[i] = 0.0;
    }
    oscil_vec[0]->evalControl_diff(t, dRedp, dImdp);
    int nparam0 = getOscillator(0)->getNParams();
    for (int iparam=0; iparam < nparam0; iparam++) {
      vals[iparam] = alpha * (coeff_p_osc0 * dRedp[iparam] + coeff_q_osc0 * dImdp[iparam]);
      cols[iparam] = iparam;
    }
    VecSetValues(grad, nparam0, cols, vals, ADD_VALUES);
    // Oscillator 1
    for (int i=0; i<nparams_max; i++){
      dRedp[i] = 0.0;
      dImdp[i] = 0.0;
    }
    oscil_vec[1]->evalControl_diff(t, dRedp, dImdp);
    int nparam1 = getOscillator(1)->getNParams();
    for (int iparam=0; iparam < nparam1; iparam++) {
      vals[iparam] = alpha * (coeff_p_osc1 * dRedp[iparam] + coeff_q_osc1 * dImdp[iparam]);
      cols[iparam] = iparam + nparam0;
    }
    VecSetValues(grad, nparam1, cols, vals, ADD_VALUES);
    VecAssemblyBegin(grad);
    VecAssemblyEnd(grad);

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

  int ilow, iupp, elemID;
  double val;
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

    case THREESTATES:

      /* Reset the initial conditions */
      VecZeroEntries(rho0);

      /* Get partitioning */
      VecGetOwnershipRange(rho0, &ilow, &iupp);

      /* Set the <iinit>'th initial state */
      if (iinit == 0) {
        // 1st initial state: rho(0)_IJ = 2(N-i+1)/(N(N+1)) Delta_IJ
        initID = 1;

        /* Iterate over diagonal elements of essential-level system */
        for (int i = 0; i<dim_ess; i++) {
          int i_full = mapEssToFull(i, nlevels, nessential);
          int diagID = getIndexReal(getVecID(i_full,i_full,dim_rho));
          double val = 2.*(dim_ess - i) / (dim_ess * (dim_ess + 1));
          if (ilow <= diagID && diagID < iupp) VecSetValue(rho0, diagID, val, INSERT_VALUES);
        }

      } else if (iinit == 1) {
        // 2nd initial state: rho(0)_IJ = 1/d
        initID = 2;
        for (int i = 0; i<dim_ess; i++) {
          int i_full = mapEssToFull(i,nlevels, nessential);
          for (int j = 0; j<dim_ess; j++) {
            double val = 1./dim_ess;
            int j_full = mapEssToFull(j,nlevels, nessential);
            int index = getIndexReal(getVecID(i_full,j_full,dim_rho));   // Re(rho_ij)
            VecSetValue(rho0, index, val, INSERT_VALUES); 
          }
        }

      } else if (iinit == 2) {
        // 3rd initial state: rho(0)_IJ = 1/d Delta_IJ
        initID = 3;

        /* Iterate over diagonal elements */
        for (int i = 0; i<dim_ess; i++) {
          int i_full = mapEssToFull(i,nlevels, nessential);
          int diagID = getIndexReal(getVecID(i_full,i_full,dim_rho));
          double val = 1./ dim_ess;
          if (ilow <= diagID && diagID < iupp) VecSetValue(rho0, diagID, val, INSERT_VALUES);
        }

      } else {
        printf("ERROR: Wrong initial condition setting!\n");
        exit(1);
      }

      /* Assemble rho0 */
      VecAssemblyBegin(rho0); VecAssemblyEnd(rho0);

      break;


    case DIAGONAL:
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

    case BASIS:

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
        int elemID = getIndexReal(getVecID(k, k, dim_rho)); // real part in vectorized system
        double val = 1.0;
        if (ilow <= elemID && elemID < iupp) VecSetValues(rho0, 1, &elemID, &val, INSERT_VALUES);
      } else {
      //   /* B_{kj} contains four non-zeros, two per row */
        int* rows = new int[4];
        double* vals = new double[4];

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


/* Define the action of RHS on a vector x */
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


  /* Control terms and dipole-dipole coupling terms */
  int id_kl = 0; // index for accessing Ad_kl inside Ad_vec
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

    // Coupling terms
    for (int josc=iosc+1; josc<shellctx->nlevels.size(); josc++){

      double etakl = shellctx->eta[id_kl];
      double coskl = cos(etakl*2*M_PI * shellctx->time);
      double sinkl = sin(etakl*2*M_PI * shellctx->time);
      double Jkl = shellctx->Jkl[id_kl]*2*M_PI; 
      // uout += J_kl*sin*Adklu
      MatMult((*(shellctx->Ad_vec))[id_kl], u, *shellctx->Adklu);
      VecAXPY(uout, Jkl*sinkl, *shellctx->Adklu);
      // uout += -Jkl*cos*Bdklv
      MatMult((*(shellctx->Bd_vec))[id_kl], v, *shellctx->Bdklv);
      VecAXPY(uout, -Jkl*coskl, *shellctx->Bdklv);
      // vout += Jkl*cos*Bdklu
      MatMult((*(shellctx->Bd_vec))[id_kl], u, *shellctx->Bdklu);
      VecAXPY(vout, Jkl*coskl, *shellctx->Bdklu);
      //vout += Jkl*sin*Adklv
      MatMult((*(shellctx->Ad_vec))[id_kl], v, *shellctx->Adklv);
      VecAXPY(vout, Jkl*sinkl, *shellctx->Adklv);
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


/* Define the action of RHS^T on a vector x */
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

    // Coupling terms
    for (int josc=iosc+1; josc<shellctx->nlevels.size(); josc++){

      double etakl = shellctx->eta[id_kl];
      double coskl = cos(etakl*2*M_PI * shellctx->time);
      double sinkl = sin(etakl*2*M_PI * shellctx->time);
      double Jkl = shellctx->Jkl[id_kl]*2*M_PI; 
      // uout += J_kl*sin*Adklu^T
      MatMultTranspose((*(shellctx->Ad_vec))[id_kl], u, *shellctx->Adklu);
      VecAXPY(uout, Jkl*sinkl, *shellctx->Adklu);
      // uout += +Jkl*cos*Bdklv^T
      MatMultTranspose((*(shellctx->Bd_vec))[id_kl], v, *shellctx->Bdklv);
      VecAXPY(uout,  Jkl*coskl, *shellctx->Bdklv);
      // vout += - Jkl*cos*Bdklu^T
      MatMultTranspose((*(shellctx->Bd_vec))[id_kl], u, *shellctx->Bdklu);
      VecAXPY(vout, - Jkl*coskl, *shellctx->Bdklu);
      //vout += Jkl*sin*Adklv^T
      MatMultTranspose((*(shellctx->Ad_vec))[id_kl], v, *shellctx->Adklv);
      VecAXPY(vout, Jkl*sinkl, *shellctx->Adklv);
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


/* Define the action of RHS on a vector x */
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
  double xi0  = shellctx->selfker[0];
  double xi1  = shellctx->selfker[1];   
  double xi01 = shellctx->crossker[0];  // zz-coupling
  double J01  = shellctx->Jkl[0]*2.*M_PI;  // dipole-dipole coupling
  double eta01 = shellctx->eta[0];
  double detuning_freq0 = shellctx->detuning_freq[0];
  double detuning_freq1 = shellctx->detuning_freq[1];
  double decay0 = 0.0;
  double decay1 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  if (shellctx->collapse_time[0] > 1e-14 && shellctx->addT1)
    decay0 = 1./shellctx->collapse_time[0];
  if (shellctx->collapse_time[1] > 1e-14 && shellctx->addT2)
    dephase0 = 1./shellctx->collapse_time[1];
  if (shellctx->collapse_time[2] > 1e-14 && shellctx->addT1)
    decay1= 1./shellctx->collapse_time[2];
  if (shellctx->collapse_time[3] > 1e-14 && shellctx->addT2)
    dephase1 = 1./shellctx->collapse_time[3];
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double cos01 = cos(eta01*2*M_PI * shellctx->time);
  double sin01 = sin(eta01*2*M_PI * shellctx->time);

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


          /* --- Offdiagonal: coupling term, oscil 0<->1 --- */
          //  1) J_kl (-icos + sin) * ρ_{E−k+l i, i′}
          if (i0 > 0 && i1 < n1-1) {
            int itx = it - stridei0 + stridei1;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0 * (i1 + 1));
            // sin u + cos v + i ( -cos u + sin v)
            yre += J01 * sq * (   cos01 * xim + sin01 * xre);
            yim += J01 * sq * ( - cos01 * xre + sin01 * xim);
          }
          // 2) J_kl (−icos − sin)sqrt(il*(ik +1)) ρ_{E+k−li,i′}
          if (i0 < n0-1 && i1 > 0) {
            int itx = it + stridei0 - stridei1;  // E+k-l i, i'
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1 * (i0 + 1)); // sqrt( il*(ik+1))
            // -sin u + cos v + i (-cos u - sin v)
            yre += J01 * sq * (   cos01 * xim - sin01 * xre);
            yim += J01 * sq * ( - cos01 * xre - sin01 * xim);
          }
          // 3) J_kl ( icos + sin)sqrt(ik'*(il' +1)) ρ_{i,E-k+li'}
          if (i0p > 0 && i1p < n1-1) {
            int itx = it - stridei0p + stridei1p;  // i, E-k+l i'
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0p * (i1p + 1)); // sqrt( ik'*(il'+1))
            //  sin u - cos v + i ( cos u + sin v)
            yre += J01 * sq * ( - cos01 * xim + sin01 * xre);
            yim += J01 * sq * (   cos01 * xre + sin01 * xim);
          }
          // 4) J_kl ( icos - sin)sqrt(il'*(ik' +1)) ρ_{i,E+k-li'}
          if (i0p < n0-1 && i1p > 0) {
            int itx = it + stridei0p - stridei1p;  // i, E+k-l i'
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1p * (i0p + 1)); // sqrt( il'*(ik'+1))
            // - sin u - cos v + i ( cos u - sin v)
            yre += J01 * sq * ( - cos01 * xim - sin01 * xre);
            yim += J01 * sq * (   cos01 * xre - sin01 * xim);
          }


          /* --- Offdiagonal part of decay L1 */
          // Oscillators 0
          if (i0 < n0-1 && i0p < n0-1) {
            double l1off = decay0 * sqrt((i0+1)*(i0p+1));
            int itx = it + stridei0 + stridei0p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            yre += l1off * xre;
            yim += l1off * xim;
          }
          // Oscillator 1
          if (i1 < n1-1 && i1p < n1-1) {
            double l1off = decay1 * sqrt((i1+1)*(i1p+1));
            int itx = it + stridei1 + stridei1p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            yre += l1off * xre;
            yim += l1off * xim;
          }

          /* --- Control hamiltonian --- Oscillator 0 --- */
          /* \rho(ik+1..,ik'..) term */
          if (i0 < n0-1) {
            int itx = it + stridei0;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0 + 1);
            yre += sq * (   pt0 * xim + qt0 * xre);
            yim += sq * ( - pt0 * xre + qt0 * xim);
          }
          /* \rho(ik..,ik'+1..) */
          if (i0p < n0-1) {
            int itx = it + stridei0p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0p + 1);
            yre += sq * ( -pt0 * xim + qt0 * xre);
            yim += sq * (  pt0 * xre + qt0 * xim);
          }
          /* \rho(ik-1..,ik'..) */
          if (i0 > 0) {
            int itx = it - stridei0;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0);
            yre += sq * (  pt0 * xim - qt0 * xre);
            yim += sq * (- pt0 * xre - qt0 * xim);
          }
          /* \rho(ik..,ik'-1..) */
          if (i0p > 0) {
            int itx = it - stridei0p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0p);
            yre += sq * (- pt0 * xim - qt0 * xre);
            yim += sq * (  pt0 * xre - qt0 * xim);
          }

          /* --- Control hamiltonian --- Oscillator 1 --- */
          /* \rho(ik+1..,ik'..) term */
          if (i1 < n1-1) {
            int itx = it + stridei1;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1 + 1);
            yre += sq * (   pt1 * xim + qt1 * xre);
            yim += sq * ( - pt1 * xre + qt1 * xim);
          }
          /* \rho(ik..,ik'+1..) */
          if (i1p < n1-1) {
            int itx = it + stridei1p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1p + 1);
            yre += sq * ( -pt1 * xim + qt1 * xre);
            yim += sq * (  pt1 * xre + qt1 * xim);
          }
          /* \rho(ik-1..,ik'..) */
          if (i1 > 0) {
            int itx = it - stridei1;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1);
            yre += sq * (  pt1 * xim - qt1 * xre);
            yim += sq * (- pt1 * xre - qt1 * xim);
          }
          /* \rho(ik..,ik'-1..) */
          if (i1p > 0) {
            /* Get output index in vectorized, colocated y */
            int itx = it - stridei1p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1p);
            yre += sq * (- pt1 * xim - qt1 * xre);
            yim += sq * (  pt1 * xre - qt1 * xim);
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

/* Define the action of RHS^T on a vector x */
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
  double xi0  = shellctx->selfker[0];
  double xi1  = shellctx->selfker[1];
  double xi01 = shellctx->crossker[0];  // zz-coupling 
  double J01 = shellctx->Jkl[0]*2.*M_PI;   // dipole-dipole coupling
  double eta01 = shellctx->eta[0];
  double detuning_freq0 = shellctx->detuning_freq[0];
  double detuning_freq1 = shellctx->detuning_freq[1];
  double decay0 = 0.0;
  double decay1 = 0.0;
  double dephase0= 0.0;
  double dephase1= 0.0;
  if (shellctx->collapse_time[0] > 1e-14 && shellctx->addT1)
    decay0 = 1./shellctx->collapse_time[0];
  if (shellctx->collapse_time[1] > 1e-14 && shellctx->addT2)
    dephase0 = 1./shellctx->collapse_time[1];
  if (shellctx->collapse_time[2] > 1e-14 && shellctx->addT1)
    decay1= 1./shellctx->collapse_time[2];
  if (shellctx->collapse_time[3] > 1e-14 && shellctx->addT2)
    dephase1 = 1./shellctx->collapse_time[3];
  double pt0 = shellctx->control_Re[0];
  double qt0 = shellctx->control_Im[0];
  double pt1 = shellctx->control_Re[1];
  double qt1 = shellctx->control_Im[1];
  double cos01 = cos(eta01*2*M_PI * shellctx->time);
  double sin01 = sin(eta01*2*M_PI * shellctx->time);

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

          /* --- Offdiagonal coupling term J_01, oscil 0<->1 --- */
          //  1) [...] * \bar y_{E+k-l i, i′}
          if (i0 < n0-1 && i1 > 0) {
            int itx = it + stridei0 - stridei1;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1 * (i0 + 1));
            yre += J01 * sq * ( - cos01 * xim + sin01 * xre);
            yim += J01 * sq * ( + cos01 * xre + sin01 * xim);
          }
          // 2) J_kl (−icos − sin)sqrt(ik*(il +1)) \bar y_{E-k+li,i′}
          if (i0 > 0 && i1 < n1-1) {
            int itx = it - stridei0 + stridei1;  // E-k+l i, i'
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0 * (i1 + 1)); // sqrt( ik*(il+1))
            yre += J01 * sq * ( - cos01 * xim - sin01 * xre);
            yim += J01 * sq * ( + cos01 * xre - sin01 * xim);
          }
          // 3) J_kl ( icos + sin)sqrt(il'*(ik' +1)) \bar y_{i,E+k-li'}
          if (i0p < n0-1 && i1p > 0) {
            int itx = it + stridei0p - stridei1p;  // i, E+k-l i'
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1p * (i0p + 1)); // sqrt( il'*(ik'+1))
            yre += J01 * sq * (   cos01 * xim + sin01 * xre);
            yim += J01 * sq * ( - cos01 * xre + sin01 * xim);
          }
          // 4) J_kl ( icos - sin)sqrt(ik'*(il' +1)) \bar y_{i,E-k+li'}
          if (i0p > 0 && i1p < n1-1) {
            int itx = it - stridei0p + stridei1p;  // i, E-k+l i'
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0p * (i1p + 1)); // sqrt( ik'*(il'+1))
            yre += J01 * sq * (   cos01 * xim - sin01 * xre);
            yim += J01 * sq * ( - cos01 * xre - sin01 * xim);
          }

          /* --- Offdiagonal part of decay L1^T */
          // Oscillators 0
          if (i0 > 0 && i0p > 0) {
            double l1off = decay0 * sqrt(i0*i0p);
            int itx = it - stridei0 - stridei0p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            yre += l1off * xre;
            yim += l1off * xim;
          }
          // Oscillator 1
          if (i1 > 0 && i1p > 0) {
            double l1off = decay1 * sqrt(i1*i1p);
            int itx = it - stridei1 - stridei1p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            yre += l1off * xre;
            yim += l1off * xim;
          }

          /* --- Control hamiltonian --- Oscillator 0 --- */
          /* \rho(ik+1..,ik'..) term */
          if (i0 > 0) {
            int itx = it - stridei0;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0);
            yre += sq * ( - pt0 * xim + qt0 * xre);
            yim += sq * (   pt0 * xre + qt0 * xim);
          }
          /* \rho(ik..,ik'+1..) */
          if (i0p > 0) {
            int itx = it - stridei0p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0p);
            yre += sq * (  pt0 * xim + qt0 * xre);
            yim += sq * ( -pt0 * xre + qt0 * xim);
          }
          /* \rho(ik-1..,ik'..) */
          if (i0 < n0-1) {
            int itx = it + stridei0;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0+1);
            yre += sq * (- pt0 * xim - qt0 * xre);
            yim += sq * (  pt0 * xre - qt0 * xim);
          }
          /* \rho(ik..,ik'-1..) */
          if (i0p < n0-1) {
            int itx = it + stridei0p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i0p+1);
            yre += sq * (+ pt0 * xim - qt0 * xre);
            yim += sq * (- pt0 * xre - qt0 * xim);
          }

          /* --- Control hamiltonian --- Oscillator 1 --- */
          /* \rho(ik+1..,ik'..) term */
          if (i1 > 0) {
            int itx = it - stridei1;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1);
            yre += sq * ( - pt1 * xim + qt1 * xre);
            yim += sq * (   pt1 * xre + qt1 * xim);
          }
          /* \rho(ik..,ik'+1..) */
          if (i1p > 0) {
            int itx = it - stridei1p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1p);
            yre += sq * (  pt1 * xim + qt1 * xre);
            yim += sq * ( -pt1 * xre + qt1 * xim);
          }
          /* \rho(ik-1..,ik'..) */
          if (i1 < n1-1) {
            int itx = it + stridei1;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1+1);
            yre += sq * (- pt1 * xim - qt1 * xre);
            yim += sq * (  pt1 * xre - qt1 * xim);
          }
          /* \rho(ik..,ik'-1..) */
          if (i1p < n1-1) {
            /* Get output index in vectorized, colocated y */
            int itx = it + stridei1p;
            double xre = xptr[2 * itx];
            double xim = xptr[2 * itx + 1];
            double sq = sqrt(i1p+1);
            yre += sq * (  pt1 * xim - qt1 * xre);
            yim += sq * (- pt1 * xre - qt1 * xim);
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


int myMatMult_matfree_2osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);


  int n0 = shellctx->nlevels[0];
  int n1 = shellctx->nlevels[1];
  if(n0==3 && n1==20){
    return myMatMult_matfree<3,20>(RHS, x, y);
  } else if(n0==3 && n1==10){
    return myMatMult_matfree<3,10>(RHS, x, y);
  } else if(n0==4 && n1==4){
    return myMatMult_matfree<4,4>(RHS, x, y);
  } else if(n0==1 && n1==1){
    return myMatMult_matfree<1,1>(RHS, x, y);
  } else if(n0==2 && n1==2){
    return myMatMult_matfree<2,2>(RHS, x, y);
  } else if(n0==3 && n1==3){
    return myMatMult_matfree<3,3>(RHS, x, y);
  } else if(n0==20 && n1==20){
    return myMatMult_matfree<20,20>(RHS, x, y);
  } else {
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
  if(n0==3 && n1==20){
    return myMatMultTranspose_matfree<3,20>(RHS, x, y);
  } else if(n0==3 && n1==10){
    return myMatMultTranspose_matfree<3,10>(RHS, x, y);
  } else if(n0==4 && n1==4){
    return myMatMultTranspose_matfree<4,4>(RHS, x, y);
  } else if(n0==1 && n1==1){
    return myMatMultTranspose_matfree<1,1>(RHS, x, y);
  } else if(n0==2 && n1==2){
    return myMatMultTranspose_matfree<2,2>(RHS, x, y);
  } else if(n0==3 && n1==3){
    return myMatMultTranspose_matfree<3,3>(RHS, x, y);
  } else if(n0==20 && n1==20){
    return myMatMultTranspose_matfree<20,20>(RHS, x, y);
  } else {
    printf("ERROR: In order to run this case, add a line at the end of mastereq.cpp with the corresponding number of levels!\n");
    exit(1);
  }
}


