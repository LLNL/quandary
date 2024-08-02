#include "mastereq.hpp"

MasterEq::MasterEq(){
  dim = 0;
  oscil_vec = NULL;
  RHS = NULL;
  Ad     = NULL;
  Bd     = NULL;
  dRedp = NULL;
  dImdp = NULL;
  usematfree = false;
  useUDEmodel = false;
  quietmode = false;
}


MasterEq::MasterEq(std::vector<int> nlevels_, std::vector<int> nessential_, Oscillator** oscil_vec_, const std::vector<double> crosskerr_, const std::vector<double> Jkl_, const std::vector<double> eta_, LindbladType lindbladtype_, bool usematfree_, bool useUDEmodel_, bool x_is_control_, Learning* learning_, std::string hamiltonian_file_, bool quietmode_) {
  int ierr;

  nlevels = nlevels_;
  nessential = nessential_;
  noscillators = nlevels.size();
  oscil_vec = oscil_vec_;
  crosskerr = crosskerr_;
  Jkl = Jkl_;
  eta = eta_;
  usematfree = usematfree_;
  useUDEmodel = useUDEmodel_;
  x_is_control = x_is_control_;
  learning = learning_;
  lindbladtype = lindbladtype_;
  hamiltonian_file = hamiltonian_file_;
  quietmode = quietmode_;


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
  if (lindbladtype != LindbladType::NONE) {  // Solve Lindblads equation, dim = N^2
    dim = dim_rho*dim_rho; 
    if (mpirank_world == 0 && !quietmode) {
      printf("Solving Lindblads master equation (open quantum system).\n");
    }
  } else { // Solve Schroedingers equation. dim = N
    dim = dim_rho; 
    if (mpirank_world == 0 && !quietmode) {
      printf("Solving Schroedingers equation (closed quantum system).\n");
    }
  }

  /* Sanity check for parallel petsc */
  if (dim % mpisize_petsc != 0) {
    if (mpirank_world==0) printf("\n ERROR in parallel distribution: Petsc's communicator size (%d) must be integer divisor of system dimension (%d).\n", mpisize_petsc, dim);
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
      addT1 = false;
      addT2 = false;
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


  /* Create transfer functions for controls, default: one per oscillator being the identity. If python interface: could be more */
  for (int k=0; k<noscillators; k++){
    IdentityTransferFunction* mytransfer_re = new IdentityTransferFunction();
    IdentityTransferFunction* mytransfer_im = new IdentityTransferFunction();
    std::vector<TransferFunction*> myvec_re{mytransfer_re};
    std::vector<TransferFunction*> myvec_im{mytransfer_im};
    transfer_Hc_re.push_back(myvec_re);
    transfer_Hc_im.push_back(myvec_im);
  }

  /* Create transfer functions for time-varying system Hamiltonian */
  // By default, these are for the Jaynes Cumming coupling: Jkl*cos(eta*t)(a+adag) - i Jkl*sin(eta*t)(a-adag)
  // If python interface, they can be different
  for (int k=0; k<noscillators*(noscillators-1)/2; k++){
    if (fabs(Jkl[k]) > 1e-12) {
      CosineTransferFunction* mytransfer_re = new CosineTransferFunction(1.0, eta[k]);
      SineTransferFunction* mytransfer_im = new SineTransferFunction(1.0, eta[k]);

      transfer_Hdt_re.push_back(mytransfer_re);
      transfer_Hdt_im.push_back(mytransfer_im);
    }
  }

  /* Initialize Hamiltonian matrices */
  if (!usematfree) {
    initSparseMatSolver();
  } 

  /* Create vector strides for accessing Re and Im part in x */
  PetscInt ilow, iupp;
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
  cols = new PetscInt[nparams_max];
  vals = new PetscScalar[nparams_max];

  /* Allocate MatShell context for applying RHS */
  RHSctx.isu = &isu;
  RHSctx.isv = &isv;
  RHSctx.crosskerr = crosskerr;
  RHSctx.Jkl = Jkl;
  RHSctx.eta = eta;
  RHSctx.addT1 = addT1;
  RHSctx.addT2 = addT2;
  RHSctx.lindbladtype = lindbladtype;
  if (!usematfree){
    RHSctx.Ac_vec = Ac_vec;
    RHSctx.Bc_vec = Bc_vec;
    RHSctx.Ad_vec = Ad_vec;
    RHSctx.Bd_vec = Bd_vec;
    RHSctx.Ad = &Ad;
    RHSctx.Bd = &Bd;
    RHSctx.aux = &aux;
  }
  RHSctx.learning = learning;
  RHSctx.useUDEmodel= useUDEmodel;
  RHSctx.nlevels = nlevels;
  RHSctx.oscil_vec = oscil_vec;
  RHSctx.time = 0.0;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    std::vector<double> controlRek;
    for (int icon=0; icon<transfer_Hc_re[iosc].size(); icon++){ 
     controlRek.push_back(0.0);
    }
    RHSctx.control_Re.push_back(controlRek);
    std::vector<double> controlImk;
    for (int icon=0; icon<transfer_Hc_im[iosc].size(); icon++){ 
     controlImk.push_back(0.0);
    }
    RHSctx.control_Im.push_back(controlImk);
  }
  for (int kl = 0; kl<transfer_Hdt_re.size(); kl++) RHSctx.eval_transfer_Hdt_re.push_back(0.0);
  for (int kl = 0; kl<transfer_Hdt_im.size(); kl++) RHSctx.eval_transfer_Hdt_im.push_back(0.0);

  /* Set the MatMult routine for applying the RHS to a vector x */
  if (usematfree) { // matrix-free solver
    if (noscillators == 1) {
      MatShellSetOperation(RHS, MATOP_MULT, (void(*)(void)) myMatMult_matfree_1Osc);
      MatShellSetOperation(RHS, MATOP_MULT_TRANSPOSE, (void(*)(void)) myMatMultTranspose_matfree_1Osc);
    } else if (noscillators == 2) {
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
      printf("ERROR. Matfree solver only for up to 5 oscillators. This should never happen! %d\n", noscillators);
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
      for (int k=0; k<Ad_vec.size(); k++) {
        if (Ad_vec[k] != NULL) {
          MatDestroy(&(Ad_vec[k]));
        }
      }
      for (int k=0; k<Bd_vec.size(); k++) {
        if (Bd_vec[k] != NULL) {
          MatDestroy(&(Bd_vec[k]));
        }
      }
      VecDestroy(&aux);
      for (int i=0; i<Ac_vec.size(); i++){
        for (int icon=0; icon<Ac_vec[i].size(); icon++)  {
          if (Ac_vec[i][icon] != NULL) {
            MatDestroy(&(Ac_vec[i][icon]));
          }
        }
      }
      for (int i=0; i<Bc_vec.size(); i++){
        for (int icon=0; icon<Bc_vec[i].size(); icon++)  {
          if (Bc_vec[i][icon] != NULL) {
            MatDestroy(&(Bc_vec[i][icon]));
          }
        }
      }
    }
    for (int i=0; i<transfer_Hdt_re.size(); i++) delete transfer_Hdt_re[i];
    for (int i=0; i<transfer_Hdt_im.size(); i++) delete transfer_Hdt_im[i];
    for (int i=0; i<transfer_Hc_re.size(); i++) {
      for (int icon=0; icon<transfer_Hc_re[i].size(); icon++) delete transfer_Hc_re[i][icon];
    }
    for (int i=0; i<transfer_Hc_im.size(); i++) {
      for (int icon=0; icon<transfer_Hc_im[i].size(); icon++) delete transfer_Hc_im[i][icon];
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

  /* Allocate all system matrices */

  // Time-independent system Hamiltonian
  // Ad = real(-i Hsys) and Bd = imag(-i Hsys)
  MatCreate(PETSC_COMM_WORLD, &Ad);
  MatCreate(PETSC_COMM_WORLD, &Bd);
  MatSetSizes(Ad, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatSetSizes(Bd, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatSetType(Ad, MATMPIAIJ);
  MatSetType(Bd, MATMPIAIJ);
  if (addT1 || addT2) MatMPIAIJSetPreallocation(Ad, noscillators+5, NULL, noscillators+5, NULL);
  MatMPIAIJSetPreallocation(Bd, 1, NULL, 1, NULL);
  MatSetUp(Ad);
  MatSetUp(Bd);
  MatSetFromOptions(Ad);
  MatSetFromOptions(Bd);
  // Allow for setting new nonzeros entries into Ad after pre-allocation
  MatSetOption(Ad, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  // One control operator per oscillator
  // Ac_vec[0] = real(-i Hc) and Bc_vec[0] = imag(-i Hc)
  for (int iosc = 0; iosc < noscillators; iosc++) {
    Mat myAcMatk, myBcMatk;
    std::vector<Mat> myAcvec_k{myAcMatk};   
    std::vector<Mat> myBcvec_k{myBcMatk};
    Ac_vec.push_back(myAcvec_k);
    Bc_vec.push_back(myBcvec_k);
    MatCreate(PETSC_COMM_WORLD, &(Ac_vec[iosc][0]));
    MatCreate(PETSC_COMM_WORLD, &(Bc_vec[iosc][0]));
    MatSetType(Ac_vec[iosc][0], MATMPIAIJ);
    MatSetType(Bc_vec[iosc][0], MATMPIAIJ);
    MatSetSizes(Ac_vec[iosc][0], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatSetSizes(Bc_vec[iosc][0], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    if (lindbladtype != LindbladType::NONE) {
      MatMPIAIJSetPreallocation(Ac_vec[iosc][0], 4, NULL, 4, NULL);
      MatMPIAIJSetPreallocation(Bc_vec[iosc][0], 4, NULL, 4, NULL);
    } else {
      MatMPIAIJSetPreallocation(Ac_vec[iosc][0], 2, NULL, 2, NULL);
      MatMPIAIJSetPreallocation(Bc_vec[iosc][0], 2, NULL, 2, NULL);
    }
    MatSetUp(Ac_vec[iosc][0]);
    MatSetUp(Bc_vec[iosc][0]);
    MatSetFromOptions(Ac_vec[iosc][0]);
    MatSetFromOptions(Bc_vec[iosc][0]); 
  }
  // Time-dependent system Hamiltonian matrices (other than controls)
  int id_kl = 0;
  for (int iosc = 0; iosc < noscillators; iosc++) {
    for (int josc=iosc+1; josc<noscillators; josc++){
      if (fabs(Jkl[id_kl]) > 1e-12) { // only allocate if Jkl>0
        Mat myAdkl, myBdkl;
        Ad_vec.push_back(myAdkl);
        Bd_vec.push_back(myBdkl);
        MatCreate(PETSC_COMM_WORLD, &Ad_vec[id_kl]);
        MatCreate(PETSC_COMM_WORLD, &Bd_vec[id_kl]);
        MatSetType(Ad_vec[id_kl], MATMPIAIJ);
        MatSetType(Bd_vec[id_kl], MATMPIAIJ);
        MatSetSizes(Ad_vec[id_kl], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
        MatSetSizes(Bd_vec[id_kl], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
        if (lindbladtype != LindbladType::NONE) {
          MatMPIAIJSetPreallocation(Ad_vec[id_kl], 4, NULL, 4, NULL);
          MatMPIAIJSetPreallocation(Bd_vec[id_kl], 4, NULL, 4, NULL);
        } else {
          MatMPIAIJSetPreallocation(Ad_vec[id_kl], 2, NULL, 2, NULL);
          MatMPIAIJSetPreallocation(Bd_vec[id_kl], 2, NULL, 2, NULL);
        }
        MatSetUp(Ad_vec[id_kl]);
        MatSetUp(Bd_vec[id_kl]);
        MatSetFromOptions(Ad_vec[id_kl]);
        MatSetFromOptions(Bd_vec[id_kl]);
      }
      id_kl++;
    }
  }

  int dimmat = dim_rho; // this is N!

  PetscInt ilow, iupp;
  int r1,r2, r1a, r2a, r1b, r2b;
  int col, col1, col2;
  double val;

  /* If a Hamiltonian file is given, read the system matrices from file. */ 
  if (hamiltonian_file.compare("none") != 0 ) {
    if (mpirank_world==0 && !quietmode) printf("\n# Reading Hamiltonian model from file %s.\n\n", hamiltonian_file.c_str());

    /* Read Hamiltonians from file */
    PythonInterface* py = new PythonInterface(hamiltonian_file, lindbladtype, dim_rho, quietmode);
    py->receiveHsys(Bd, Ad);
    py->receiveHc(noscillators, Ac_vec, Bc_vec); 

    if (mpirank_world==0&& !quietmode) printf("# Done. \n\n");
    delete py;

  /* Else: Initialize Hamiltonian system matrices with standard Hamiltonian model */
  } else {

    for (int iosc = 0; iosc < noscillators; iosc++) {

      /* Get dimensions */
      int nk     = oscil_vec[iosc]->getNLevels();
      int nprek  = oscil_vec[iosc]->dim_preOsc;
      int npostk = oscil_vec[iosc]->dim_postOsc;

      /* Set control Hamiltonian system matrix real(-iHc) */
      /* Lindblad solver:     Ac = I_N \kron (a - a^T) - (a - a^T)^T \kron I_N   \in C^{N^2 x N^2}*/
      /* Schroedinger solver: Ac = a - a^T   \in C^{N x N}  */
      MatGetOwnershipRange(Ac_vec[iosc][0], &ilow, &iupp);
      for (int row = ilow; row<iupp; row++){
        // A_c or I_N \kron A_c
        col1 = row + npostk;
        col2 = row - npostk;
        if (lindbladtype != LindbladType::NONE) r1 = row % dimmat;   // I_N \kron A_c 
        else r1 = row;   // A_c
        r1 = r1 % (nk*npostk);
        r1 = r1 / npostk;
        if (r1 < nk-1) {
          val = sqrt(r1+1);
          if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc][0], row, col1, val, ADD_VALUES);
        }
        if (r1 > 0) {
          val = -sqrt(r1);
          if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc][0], row, col2, val, ADD_VALUES);
        } 
        if (lindbladtype != LindbladType::NONE){
          //- A_c \kron I_N
          col1 = row + npostk*dimmat;
          col2 = row - npostk*dimmat;
          r1 = row % (dimmat * nk * npostk);
          r1 = r1 / (dimmat * npostk);
          if (r1 < nk-1) {
            val =  sqrt(r1+1);
            if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc][0], row, col1, val, ADD_VALUES);
          }
          if (r1 > 0) {
            val = -sqrt(r1);
            if (fabs(val)>1e-14) MatSetValue(Ac_vec[iosc][0], row, col2, val, ADD_VALUES);
          }
        }
      }

      /* Set control Hamiltonian system matrix Bc=imag(-iHc) */
      /* Lindblas solver Bc = - I_N \kron (a + a^T) + (a + a^T)^T \kron I_N */
      /* Schroedinger solver: Bc = -(a+a^T) */
      /* Iterate over local rows of Bc_vec */
      MatGetOwnershipRange(Bc_vec[iosc][0], &ilow, &iupp);
      for (int row = ilow; row<iupp; row++){
        // B_c or  I_n \kron B_c 
        col1 = row + npostk;
        col2 = row - npostk;
        if (lindbladtype != LindbladType::NONE) r1 = row % dimmat; // I_n \kron B_c
        else r1 = row;  // -Bc
        r1 = r1 % (nk*npostk);
        r1 = r1 / npostk;
        if (r1 < nk-1) {
          val = -sqrt(r1+1);
          if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc][0], row, col1, val, ADD_VALUES);
        }
        if (r1 > 0) {
          val = -sqrt(r1);
          if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc][0], row, col2, val, ADD_VALUES);
        } 
        if (lindbladtype != LindbladType::NONE){
          //+ B_c \kron I_N
          col1 = row + npostk*dimmat;
          col2 = row - npostk*dimmat;
          r1 = row % (dimmat * nk * npostk);
          r1 = r1 / (dimmat * npostk);
          if (r1 < nk-1) {
            val =  sqrt(r1+1);
            if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc][0], row, col1, val, ADD_VALUES);
          }
          if (r1 > 0) {
            val = sqrt(r1);
            if (fabs(val)>1e-14) MatSetValue(Bc_vec[iosc][0], row, col2, val, ADD_VALUES);
          }   
        }
      }
    }

    /* Set Jaynes-Cummings coupling system Hamiltonian */
    /* Lindblad solver: 
     * Ad_kl(t) =  I_N \kron (ak^Tal − akal^T) − (al^Tak − alak^T) \kron IN 
     * Bd_kl(t) = -I_N \kron (ak^Tal + akal^T) + (al^Tak + alak_T) \kron IN */
    /* Schrodinger solver:
       Ad_kl(t) =  (ak^Tal - akal^T)
       Bd_kl(t) = -(ak^Tal + akal^T)  */
    id_kl=0;
    for (int iosc = 0; iosc < noscillators; iosc++) {
      // Dimensions of ioscillator
      int nk     = oscil_vec[iosc]->getNLevels();
      int nprek  = oscil_vec[iosc]->dim_preOsc;
      int npostk = oscil_vec[iosc]->dim_postOsc;

      for (int josc=iosc+1; josc<noscillators; josc++){
        if (fabs(Jkl[id_kl]) > 1e-12) { // only allocate if coefficient is non-zero to save memory.
          // Dimensions of joscillator
          int nj     = oscil_vec[josc]->getNLevels();
          int nprej  = oscil_vec[josc]->dim_preOsc;
          int npostj = oscil_vec[josc]->dim_postOsc;

          /* Iterate over local rows of Ad_vec / Bd_vec */
          MatGetOwnershipRange(Ad_vec[id_kl], &ilow, &iupp);
          for (int row = ilow; row<iupp; row++){
            // Add +/- I_N \kron (ak^Tal -/+ akal^T) (Lindblad)
            // or  +/- (ak^Tal -/+ akal^T) (Schrodinger)
            r1 = row % (dimmat / nprek);
            r1a = (int) r1 / npostk;
            r1b = r1 % (nj*npostj);
            r1b = r1b % (nj*npostj);
            r1b = (int) r1b / npostj;
            if (r1a > 0 && r1b < nj-1) {
              val = Jkl[id_kl] * sqrt(r1a * (r1b+1));
              col = row - npostk + npostj;
               if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col,  val, ADD_VALUES);
               if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, -val, ADD_VALUES);
            }
            if (r1a < nk-1  && r1b > 0) {
              val = Jkl[id_kl] * sqrt((r1a+1) * r1b);
              col = row + npostk - npostj;
              if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col, -val, ADD_VALUES);
              if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, -val, ADD_VALUES);
            }

            if (lindbladtype != LindbladType::NONE) {
              // Add -/+ (al^Tak -/+ alak^T) \kron I
              r1 = row % (dimmat * dimmat / nprek );
              r1a = (int) r1 / (npostk*dimmat);
              r1b = r1 % (npostk*dimmat);
              r1b = r1b % (nj*npostj*dimmat);
              r1b = (int) r1b / (npostj*dimmat);
              if (r1a < nk-1 && r1b > 0) {
                val = Jkl[id_kl] * sqrt((r1a+1) * r1b);
                col = row + npostk*dimmat - npostj*dimmat;
                if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col, -val, ADD_VALUES);
                if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, +val, ADD_VALUES);
              }
              if (r1a > 0 && r1b < nj-1) {
                val = Jkl[id_kl] * sqrt(r1a * (r1b+1));
                col = row - npostk*dimmat + npostj*dimmat;
                if (fabs(val)>1e-14) MatSetValue(Ad_vec[id_kl], row, col, val, ADD_VALUES);
                if (fabs(val)>1e-14) MatSetValue(Bd_vec[id_kl], row, col, val, ADD_VALUES);
              }
            }
          }
        }
        id_kl++;
      }
    }

    /* Set system Hamiltonian part Bd = imag(iHsys) */
    int coupling_id = 0;
    for (int iosc = 0; iosc < noscillators; iosc++) {

      int nk     = oscil_vec[iosc]->getNLevels();
      int nprek  = oscil_vec[iosc]->dim_preOsc;
      int npostk = oscil_vec[iosc]->dim_postOsc;
      double xik = oscil_vec[iosc]->getSelfkerr();
      double detunek = oscil_vec[iosc]->getDetuning();

      /* Diagonal: detuning and anharmonicity  */
      /* Iterate over local rows of Bd */
      MatGetOwnershipRange(Bd, &ilow, &iupp);
      for (int row = ilow; row<iupp; row++){

        // Indices for -I_N \kron B_d
        if (lindbladtype != LindbladType::NONE) r1 = row % dimmat;
        else r1 = row;
        r1 = r1 % (nk * npostk);
        r1 = (int) r1 / npostk;
        // Indices for B_d \kron I_N
        r2 = (int) row / dimmat;
        r2 = r2 % (nk * npostk);
        r2 = (int) r2 / npostk;
        if (lindbladtype == LindbladType::NONE) r2 = 0;

        // -Bd, or -I_N \kron B_d + B_d \kron I_N
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
          if (lindbladtype != LindbladType::NONE) r1 = row % dimmat;
          else r1 = row;
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
          if (lindbladtype == LindbladType::NONE) r2a = 0;
          if (lindbladtype == LindbladType::NONE) r2b = 0;

          // -I_N \kron B_d + B_d \kron I_N
          val =  xikj * r1a * r1b  - xikj * r2a * r2b;
          if (fabs(val)>1e-14) MatSetValue(Bd, row, row, val, ADD_VALUES);
        }
      }
    }
  }

  /* Assemble all system matrices */
  MatAssemblyBegin(Bd, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Bd, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Ad, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ad, MAT_FINAL_ASSEMBLY);
  id_kl = 0;
  for (int iosc = 0; iosc < noscillators; iosc++){
    MatAssemblyBegin(Ac_vec[iosc][0], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ac_vec[iosc][0], MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Bc_vec[iosc][0], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bc_vec[iosc][0], MAT_FINAL_ASSEMBLY);
    for (int josc=iosc+1; josc<noscillators; josc++){
      if (fabs(Jkl[id_kl]) > 1e-12) { // only allocate if Jkl>0
        MatAssemblyBegin(Ad_vec[id_kl], MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(Bd_vec[id_kl], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Ad_vec[id_kl], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Bd_vec[id_kl], MAT_FINAL_ASSEMBLY);
      }
      id_kl++;
    }
  }

  // Test if Bd is symmetric is symmetric 
  PetscBool isSymm;
  double norm = 0.0;
  MatIsSymmetric(Bd, 1e-12, &isSymm);
  if (!isSymm) {
    printf("ERROR: System hamiltonian is not hermitian!\n");
    exit(1);
  }
  // Test if Ad is anti-symmetric
  Mat AdTest;
  MatTranspose(Ad, MAT_INITIAL_MATRIX, &AdTest);
  MatAXPY(AdTest, 1.0, Ad, DIFFERENT_NONZERO_PATTERN);
  MatNorm(AdTest, NORM_FROBENIUS, &norm);
  if (norm > 1e-12) {
    printf("ERROR: System hamiltonian is not hermitian!\n");
    exit(1);
  }
  MatDestroy(&AdTest);
  
  /* Set Ad = Lindblad terms */
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
      MatGetOwnershipRange(Ad, &ilow, &iupp);
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
    /* Assemble Ad again */
    MatAssemblyBegin(Ad, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ad, MAT_FINAL_ASSEMBLY);
  }

    // printf("Ad =");
    // MatView(Ad, NULL);
    // printf("Bd =");
    // MatView(Bd, NULL);
    // for (int iosc =0; iosc < Ac_vec.size(); iosc++){
    //   printf("Ac %d=", iosc);
    //   MatView(Ac_vec[iosc][0], NULL);
    // }
    // for (int iosc =0; iosc < Bc_vec.size(); iosc++){
    //   printf("Bc %d=", iosc);
    //   MatView(Bc_vec[iosc][0], NULL);
    // }



  // Remove control parameters for those oscillators that are non-controllable
  for (int k=0; k<nlevels.size(); k++){
    PetscScalar norm;
    MatNorm(Ac_vec[k][0], NORM_FROBENIUS, &norm);
    if (norm < 1e-14) {
      MatDestroy(&(Ac_vec[k][0]));
      Ac_vec[k].pop_back();
    }
    MatNorm(Bc_vec[k][0], NORM_FROBENIUS, &norm);
    if (norm < 1e-14) {
      MatDestroy(&(Bc_vec[k][0]));
      Bc_vec[k].pop_back();
    }
     if (Ac_vec[k].size() == 0 && Bc_vec[k].size() == 0) getOscillator(k)->clearParams();
  }




//   // Test: Print out Hamiltonian terms.
//   printf("\n\n HEYHEY! Printing out the system matrices: \n\n");
//   printf("Ad=\n");
//   MatView(Ad, NULL);
//   printf("Bd=\n");
//   MatView(Bd, NULL);
//   for (int k=0; k<noscillators; k++){
//     for (int i=0; i<Bc_vec[k].size(); i++){
//       printf("Oscil %d, control term Bc %d:\n", k, i);
//       MatView(Bc_vec[k][i], NULL);
//     }
//     for (int i=0; i<Ac_vec[k].size(); i++){
//       printf("Oscil %d, control term Ac %d:\n", k, i);
//       MatView(Ac_vec[k][i], NULL);
//     }
//   }
//   for (int kl=0; kl<Ad_vec.size(); kl++) {
//     printf("Bd_vec[%d]=\n", kl);
//     MatView(Bd_vec[kl], NULL);
//     printf("Ad_vec[%d]=\n", kl);
//     MatView(Ad_vec[kl], NULL);
//   }
//  // exit(1);


  /* Allocate some auxiliary vectors */
  MatCreateVecs(Bd, &aux, NULL);
}

int MasterEq::getDim(){ return dim; }

int MasterEq::getDimEss(){ return dim_ess; }

int MasterEq::getDimRho(){ return dim_rho; }

int MasterEq::getNOscillators() { return noscillators; }

Oscillator* MasterEq::getOscillator(const int i) { return oscil_vec[i]; }

int MasterEq::assemble_RHS(const double t){
  int ierr;
  /* Prepare the matrix shell to perform the action of RHS on a vector */

  // Set the time
  RHSctx.time = t;

  // Evaluate and store the controls and transfer for each oscillator and each controlterm
  for (int iosc = 0; iosc < noscillators; iosc++) {

    double p, q;
    oscil_vec[iosc]->evalControl(t, &p, &q);  // Evaluates the B-spline basis functions -> p(t,alpha), q(t,alpha)

    // Iterate over control terms for this oscillator
    for (int icon=0; icon<transfer_Hc_re[iosc].size(); icon++){
      // Get transfer functions u^k_i(p) (Default: Identity. But could be different if python interface)
      double ukip = transfer_Hc_re[iosc][icon]->eval(p, t);
      RHSctx.control_Re[iosc][icon] = ukip; 
    } 
    for (int icon=0; icon<transfer_Hc_im[iosc].size(); icon++){
      double ukiq = transfer_Hc_im[iosc][icon]->eval(q, t);
      RHSctx.control_Im[iosc][icon] = ukiq;
    } 
  } 

  // Evaluate and store transfer for time-dependent system term
  for (int kl=0; kl<transfer_Hdt_re.size(); kl++)
    // REAL part: Default trans_re = Jkl*cos(etakl*t) , or from python interface
    RHSctx.eval_transfer_Hdt_re[kl] = transfer_Hdt_re[kl]->eval(t, t); 
  for (int kl=0; kl<transfer_Hdt_im.size(); kl++)
    // IMAG part: Default trans_im = Jkl*sin(etakl*t)
    RHSctx.eval_transfer_Hdt_im[kl] = transfer_Hdt_im[kl]->eval(t, t); 

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

  if (usematfree && !useUDEmodel) {  // Matrix-free solver
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

    if (noscillators == 1) {
    /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
      int n0 = nlevels[0];
      int stridei0  = TensorGetIndex(n0, 1,0);
      int stridei0p = TensorGetIndex(n0, 0,1);
      /* Switch for Lindblad vs Schroedinger solver */
      int n0p = n0;
      if (lindbladtype == LindbladType::NONE) { // Schroedinger
        n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
      }

      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0p; i0p++)  {
          for (int i0 = 0; i0 < n0; i0++)  {
              /* Get xbar */
              double xbarre = xbarptr[2*it];
              double xbarim = xbarptr[2*it+1];

              /* --- Oscillator 0 --- */
              dRHSdp_getcoeffs(it, n0, n0p, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
              coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
              coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;

              it++;
            }
        }
    }
    else if (noscillators == 2) {
    /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
      int n0 = nlevels[0];
      int n1 = nlevels[1];
      int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
      int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
      int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
      int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);
      /* Switch for Lindblad vs Schroedinger solver */
      int n0p = n0;
      int n1p = n1;
      if (lindbladtype == LindbladType::NONE) { // Schroedinger
        n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
        n1p = 1;
      }

      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0p; i0p++)  {
        for (int i1p = 0; i1p < n1p; i1p++)  {
          for (int i0 = 0; i0 < n0; i0++)  {
            for (int i1 = 0; i1 < n1; i1++)  {
              /* Get xbar */
              double xbarre = xbarptr[2*it];
              double xbarim = xbarptr[2*it+1];

              /* --- Oscillator 0 --- */
              dRHSdp_getcoeffs(it, n0, n0p, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
              coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
              coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
              /* --- Oscillator 1 --- */
              dRHSdp_getcoeffs(it, n1, n1p, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
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
      /* Switch for Lindblad vs Schroedinger solver */
      int n0p = n0;
      int n1p = n1;
      int n2p = n2;
      if (lindbladtype == LindbladType::NONE) { // Schroedinger
        n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
        n1p = 1;
        n2p = 1;
      }
      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0p; i0p++)  {
        for (int i1p = 0; i1p < n1p; i1p++)  {
          for (int i2p = 0; i2p < n2p; i2p++)  {
            for (int i0 = 0; i0 < n0; i0++)  {
              for (int i1 = 0; i1 < n1; i1++)  {
                for (int i2 = 0; i2 < n2; i2++)  {
                  /* Get xbar */
                  double xbarre = xbarptr[2*it];
                  double xbarim = xbarptr[2*it+1];

                  /* --- Oscillator 0 --- */
                  dRHSdp_getcoeffs(it, n0, n0p, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                  coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
                  coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
                  /* --- Oscillator 1 --- */
                  dRHSdp_getcoeffs(it, n1, n1p, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                  coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
                  coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;
                  /* --- Oscillator 2 --- */
                  dRHSdp_getcoeffs(it, n2, n2p, i2, i2p, stridei2, stridei2p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
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
      /* Switch for Lindblad vs Schroedinger solver */
      int n0p = n0;
      int n1p = n1;
      int n2p = n2;
      int n3p = n3;
      if (lindbladtype == LindbladType::NONE) { // Schroedinger
        n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
        n1p = 1;
        n2p = 1;
        n3p = 1;
      }
      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0p; i0p++)  {
        for (int i1p = 0; i1p < n1p; i1p++)  {
          for (int i2p = 0; i2p < n2p; i2p++)  {
            for (int i3p = 0; i3p < n3p; i3p++)  {
              for (int i0 = 0; i0 < n0; i0++)  {
                for (int i1 = 0; i1 < n1; i1++)  {
                  for (int i2 = 0; i2 < n2; i2++)  {
                    for (int i3 = 0; i3 < n3; i3++)  {
                      /* Get xbar */
                      double xbarre = xbarptr[2*it];
                      double xbarim = xbarptr[2*it+1];

                      /* --- Oscillator 0 --- */
                      dRHSdp_getcoeffs(it, n0, n0p, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
                      /* --- Oscillator 1 --- */
                      dRHSdp_getcoeffs(it, n1, n1p, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;
                      /* --- Oscillator 2 --- */
                      dRHSdp_getcoeffs(it, n2, n2p, i2, i2p, stridei2, stridei2p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                      coeff_p[2] += res_p_re * xbarre + res_p_im * xbarim;
                      coeff_q[2] += res_q_re * xbarre + res_q_im * xbarim;
                      /* --- Oscillator 3 --- */
                      dRHSdp_getcoeffs(it, n3, n3p, i3, i3p, stridei3, stridei3p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
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
      /* Switch for Lindblad vs Schroedinger solver */
      int n0p = n0;
      int n1p = n1;
      int n2p = n2;
      int n3p = n3;
      int n4p = n4;
      if (lindbladtype == LindbladType::NONE) { // Schroedinger
        n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
        n1p = 1;
        n2p = 1;
        n3p = 1;
        n4p = 1;
      }
      /* --- Collect coefficients for gradient --- */
      int it = 0;
      // Iterate over indices of xbar
      for (int i0p = 0; i0p < n0p; i0p++)  {
        for (int i1p = 0; i1p < n1p; i1p++)  {
          for (int i2p = 0; i2p < n2p; i2p++)  {
            for (int i3p = 0; i3p < n3p; i3p++)  {
              for (int i4p = 0; i4p < n4p; i4p++)  {
                for (int i0 = 0; i0 < n0; i0++)  {
                  for (int i1 = 0; i1 < n1; i1++)  {
                    for (int i2 = 0; i2 < n2; i2++)  {
                      for (int i3 = 0; i3 < n3; i3++)  {
                        for (int i4 = 0; i4 < n4; i4++)  {
                          /* Get xbar */
                          double xbarre = xbarptr[2*it];
                          double xbarim = xbarptr[2*it+1];

                          /* --- Oscillator 0 --- */
                          dRHSdp_getcoeffs(it, n0, n0p, i0, i0p, stridei0, stridei0p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[0] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[0] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 1 --- */
                          dRHSdp_getcoeffs(it, n1, n1p, i1, i1p, stridei1, stridei1p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[1] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[1] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 2 --- */
                          dRHSdp_getcoeffs(it, n2, n2p, i2, i2p, stridei2, stridei2p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[2] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[2] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 3 --- */
                          dRHSdp_getcoeffs(it, n3, n3p, i3, i3p, stridei3, stridei3p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
                          coeff_p[3] += res_p_re * xbarre + res_p_im * xbarim;
                          coeff_q[3] += res_q_re * xbarre + res_q_im * xbarim;
                          /* --- Oscillator 4 --- */
                          dRHSdp_getcoeffs(it, n4, n4p, i4, i4p, stridei4, stridei4p, xptr, &res_p_re, &res_p_im, &res_q_re, &res_q_im);
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

  } else {  // sparse matrix solver or learning 

  /* Get real and imaginary part from x and x_bar */
  Vec u, v, ubar, vbar;
  VecGetSubVector(x, isu, &u);
  VecGetSubVector(x, isv, &v);
  VecGetSubVector(xbar, isu, &ubar);
  VecGetSubVector(xbar, isv, &vbar);

  if (x_is_control) { // Gradient wrt control parameters 
  
    /* Loop over oscillators */
    int col_shift = 0;
    for (int iosc= 0; iosc < noscillators; iosc++){

      /* Evaluate the derivative of the control functions wrt control parameters */
      for (int i=0; i<nparams_max; i++){
        dRedp[i] = 0.0;
        dImdp[i] = 0.0;
      }
      oscil_vec[iosc]->evalControl_diff(t, dRedp, dImdp);

      // Derivative of transfer functions u^k_i(p), v^k_i(q) for all control terms i=0,..., ncontrol[k]-1
      std::vector<double> dukidp;
      std::vector<double> dukidq;
      double p, q;
      oscil_vec[iosc]->evalControl(t, &p, &q);  // Evaluates the B-spline basis functions -> p(t,alpha), q(t,alpha)
      for (int icon=0; icon<Bc_vec[iosc].size(); icon++){ // Now evaluate the derivative of transfer functions for each control term
        double dukidp_tmp = transfer_Hc_re[iosc][icon]->der(p, t); // dudp(p)
        dukidp.push_back(dukidp_tmp);
      }
      for (int icon=0; icon<Ac_vec[iosc].size(); icon++){ 
        double dukidq_tmp = transfer_Hc_im[iosc][icon]->der(q, t); // dvdq(q)
        dukidq.push_back(dukidq_tmp);
      }

      /* Compute terms in RHS(x)^T xbar */
      double uAubar = 0.0; 
      double vAvbar = 0.0;
      double vBubar = 0.0;
      double uBvbar = 0.0;
      for (int icon=0; icon<Ac_vec[iosc].size(); icon++){
        double dot;
        MatMult(Ac_vec[iosc][icon], u, aux); VecDot(aux, ubar, &dot); uAubar += dot * dukidq[icon];
        MatMult(Ac_vec[iosc][icon], v, aux); VecDot(aux, vbar, &dot); vAvbar += dot * dukidq[icon];
      }
      for (int icon=0; icon<Bc_vec[iosc].size(); icon++){
        double dot;
        MatMult(Bc_vec[iosc][icon], u, aux); VecDot(aux, vbar, &dot); uBvbar += dot * dukidp[icon];
        MatMult(Bc_vec[iosc][icon], v, aux); VecDot(aux, ubar, &dot); vBubar += dot * dukidp[icon];
      }

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

  } else { // Gradient wrt learnable parameters

    learning->dRHSdp(grad, u, v, alpha, ubar, vbar);
  }

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
  // Design storage: x = (params_oscil0, params_oscil2, ... ) 
  int shift=0;
  for (int ioscil = 0; ioscil < getNOscillators(); ioscil++) {
    /* Copy x into the oscillators parameter array. */
    getOscillator(ioscil)->setParams(ptr + shift);
    shift += getOscillator(ioscil)->getNParams();
  }
  VecRestoreArrayRead(x, &ptr);
}


/* Sparse matrix solver: Define the action of RHS on a vector x */
int myMatMult_sparsemat(Mat RHS, Vec x, Vec y){
  double p,q;

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

  // printf("Mastereq System Mats\n");
  // MatView(*shellctx->Ad, NULL);
  // MatView(*shellctx->Bd, NULL);


  /* -- Control Terms -- */
  for (int iosc = 0; iosc < shellctx->nlevels.size(); iosc++) {

    // Iterate over control terms for this oscillator
    for (int icon=0; icon<shellctx->Ac_vec[iosc].size(); icon++){
      q = shellctx->control_Im[iosc][icon];
      // uout += q^k*Acu
      MatMult(shellctx->Ac_vec[iosc][icon], u, *shellctx->aux);
      VecAXPY(uout, q, *shellctx->aux); 
      // vout += q^kAcv
      MatMult(shellctx->Ac_vec[iosc][icon], v, *shellctx->aux);
      VecAXPY(vout, q, *shellctx->aux);
    }
    for (int icon=0; icon<shellctx->Bc_vec[iosc].size(); icon++){
      p = shellctx->control_Re[iosc][icon];
      // uout -= p^kBcv
      MatMult(shellctx->Bc_vec[iosc][icon], v, *shellctx->aux);
      VecAXPY(uout, -1.*p, *shellctx->aux);
      // vout += p^kBcu
      MatMult(shellctx->Bc_vec[iosc][icon], u, *shellctx->aux);
      VecAXPY(vout, p, *shellctx->aux);
    }
  }

  /* --- Apply time-dependent system Hamiltonian --- */
  /* By default (no python interface), these are the Jayes-Cumming coupling terms */
  // REAL
  for (int id_kl = 0; id_kl<shellctx->Bd_vec.size(); id_kl++){
    double trans_re = shellctx->eval_transfer_Hdt_re[id_kl]; // Default: trans_re = Jkl*cos(etakl*t) 

    // printf("%f %f %f\n", shellctx->time, trans_re, trans_im);
    if (fabs(trans_re) > 1e-12) {
      // uout += -Jkl*cos*Bdklv
      MatMult(shellctx->Bd_vec[id_kl], v, *shellctx->aux);
      VecAXPY(uout, -trans_re, *shellctx->aux);
      // vout += Jkl*cos*Bdklu
      MatMult(shellctx->Bd_vec[id_kl], u, *shellctx->aux);
      VecAXPY(vout, trans_re, *shellctx->aux);
      
      // if (shellctx->time >= 1.0){
      //   printf("trans_re %f  %.8f\n", shellctx->time, trans_re);
      //   MatView(shellctx->Bd_vec[id_kl], NULL);
      // }
    }
  }
  // IMAG
  for (int id_kl = 0; id_kl<shellctx->Ad_vec.size(); id_kl++){
    // Get transfer function
    double trans_im = shellctx->eval_transfer_Hdt_im[id_kl]; // Default: trans_im = Jkl*sin(etakl*t)

    if (fabs(trans_im) > 1e-12) {
      // uout += J_kl*sin*Adklu
      MatMult(shellctx->Ad_vec[id_kl], u, *shellctx->aux);
      VecAXPY(uout, trans_im, *shellctx->aux);
      //vout += Jkl*sin*Adklv
      MatMult(shellctx->Ad_vec[id_kl], v, *shellctx->aux);
      VecAXPY(vout, trans_im, *shellctx->aux);

      // if (shellctx->time >= 1.0) {
      //   printf("trans_im %f  %.8f\n", shellctx->time, trans_im);
      //   MatView(shellctx->Ad_vec[id_kl], NULL);
      // }
    }
  }

  /* --- Apply learning terms --- */
  shellctx->learning->applyLearningTerms(u,v,uout, vout);
  // exit(1);

  /* Restore */
  VecRestoreSubVector(x, *shellctx->isu, &u);
  VecRestoreSubVector(x, *shellctx->isv, &v);
  VecRestoreSubVector(y, *shellctx->isu, &uout);
  VecRestoreSubVector(y, *shellctx->isv, &vout);

  return 0;
}


/* Sparse-matrix solver: Define the action of RHS^T on a vector x */
int myMatMultTranspose_sparsemat(Mat RHS, Vec x, Vec y) {
  double p,q;

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

  /* Time-dependent control term */
  for (int iosc = 0; iosc < shellctx->nlevels.size(); iosc++) {

    // Iterate over control terms for this oscillator
    for (int icon=0; icon<shellctx->Ac_vec[iosc].size(); icon++){
      q = shellctx->control_Im[iosc][icon];
      // uout += q^k*Ac^Tu
      MatMultTranspose(shellctx->Ac_vec[iosc][icon], u, *shellctx->aux);
      VecAXPY(uout, q, *shellctx->aux);
      // vout += q^kAc^Tv
      MatMultTranspose(shellctx->Ac_vec[iosc][icon], v, *shellctx->aux);
      VecAXPY(vout, q, *shellctx->aux);
    }
    for (int icon=0; icon<shellctx->Bc_vec[iosc].size(); icon++){
      p = shellctx->control_Re[iosc][icon];
      // uout += p^kBc^Tv
      MatMultTranspose(shellctx->Bc_vec[iosc][icon], v, *shellctx->aux);
      VecAXPY(uout, p, *shellctx->aux);
      // vout -= p^kBc^Tu
      MatMultTranspose(shellctx->Bc_vec[iosc][icon], u, *shellctx->aux);
      VecAXPY(vout, -1.*p, *shellctx->aux);
    }
  }


  /* --- Apply time-dependent system Hamiltonian --- */
  /* By default (no python interface), these are the Jayes-Cumming coupling terms */
  // REAL
  for (int id_kl = 0; id_kl<shellctx->Bd_vec.size(); id_kl++){
    double trans_re = shellctx->eval_transfer_Hdt_re[id_kl]; // Default: trans_re = Jkl*cos(etakl*t) 
    if (fabs(trans_re) > 1e-12) {
      // uout += +Jkl*cos*Bdklv^T
      MatMultTranspose(shellctx->Bd_vec[id_kl], v, *shellctx->aux);
      VecAXPY(uout,  trans_re, *shellctx->aux);
      // vout += - Jkl*cos*Bdklu^T
      MatMultTranspose(shellctx->Bd_vec[id_kl], u, *shellctx->aux);
      VecAXPY(vout, - trans_re, *shellctx->aux);
    }
  }
  // IMAG
  for (int id_kl = 0; id_kl<shellctx->Ad_vec.size(); id_kl++){
    // Get transfer function
    double trans_im = shellctx->eval_transfer_Hdt_im[id_kl]; // Default: trans_im = Jkl*sin(etakl*t)

    if (fabs(trans_im) > 1e-12) {
      // uout += J_kl*sin*Adklu^T
      MatMultTranspose(shellctx->Ad_vec[id_kl], u, *shellctx->aux);
      VecAXPY(uout, trans_im, *shellctx->aux);
      //vout += Jkl*sin*Adklv^T
      MatMultTranspose(shellctx->Ad_vec[id_kl], v, *shellctx->aux);
      VecAXPY(vout, trans_im, *shellctx->aux);
    }
  }

  /* --- Apply learning terms --- */
  shellctx->learning->applyLearningTerms_diff(u,v,uout, vout);

  /* Restore */
  VecRestoreSubVector(x, *shellctx->isu, &u);
  VecRestoreSubVector(x, *shellctx->isv, &v);
  VecRestoreSubVector(y, *shellctx->isu, &uout);
  VecRestoreSubVector(y, *shellctx->isv, &vout);

  return 0;
}

/* Matfree-solver for 1 Oscillator: Define the action of RHS on a vector x */
template <int n0>
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
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double decay0 = 0.0;
  double dephase0= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)
    decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2)
    dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,1,0);
  int stridei0p = TensorGetIndex(n0,0,1);

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {

          /* --- Diagonal part ---*/
          //Get input x values
          double xre = xptr[2 * it];
          double xim = xptr[2 * it + 1];
          // drift Hamiltonian: uout = ( hd(ik) - hd(ik'))*vin
          //                    vout = (-hd(ik) + hd(ik'))*uin
          double hd  = H_detune(detuning_freq0, i0)
                     + H_selfkerr(xi0, i0);
          double hdp = 0.0;
          if (shellctx->lindbladtype != LindbladType::NONE) {
            hdp = H_detune(detuning_freq0, i0p)
                + H_selfkerr(xi0, i0p);
          }
          double yre = ( hd - hdp ) * xim;
          double yim = (-hd + hdp ) * xre;

          // Decay l1, diagonal part: xout += l1diag xin
          // Dephasing l2: xout += l2(ik, ikp) xin
          if (shellctx->lindbladtype != LindbladType::NONE) {
            double l1diag = L1diag(decay0, i0, i0p);
            double l2 = L2(dephase0, i0, i0p);
            yre += (l2 + l1diag) * xre;
            yim += (l2 + l1diag) * xim;
          }

          /* --- Offdiagonal: Jkl coupling term --- */

          /* --- Offdiagonal part of decay L1 */
          if (shellctx->lindbladtype != LindbladType::NONE) {
            L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
          }

          /* --- Control hamiltonian --- */
          // Oscillator 0 
          control(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);

          /* Update */
          yptr[2*it]   = yre;
          yptr[2*it+1] = yim;
          it++;
      }
  }

  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

  return 0;
}

/* Matrix-free solver for 1 Oscillators: Define the action of RHS^T on a vector x */
template <int n0>
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
  double detuning_freq0 = shellctx->oscil_vec[0]->getDetuning();
  double decay0 = 0.0;
  double dephase0= 0.0;
  if (shellctx->oscil_vec[0]->getDecayTime() > 1e-14 && shellctx->addT1)
    decay0 = 1./shellctx->oscil_vec[0]->getDecayTime();
  if (shellctx->oscil_vec[0]->getDephaseTime() > 1e-14 && shellctx->addT2)
    dephase0 = 1./shellctx->oscil_vec[0]->getDephaseTime();
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0, 1,0);
  int stridei0p = TensorGetIndex(n0, 0,1);

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
      for (int i0 = 0; i0 < n0; i0++)  {

          /* --- Diagonal part ---*/
          //Get input x values
          double xre = xptr[2 * it];
          double xim = xptr[2 * it + 1];
          // drift Hamiltonian Hd^T: uout = ( hd(ik) - hd(ik'))*vin
          //                         vout = (-hd(ik) + hd(ik'))*uin
          double hd  = H_detune(detuning_freq0, i0)
                     + H_selfkerr(xi0, i0);
          double hdp = 0.0;
          if (shellctx->lindbladtype != LindbladType::NONE) {
            hdp = H_detune(detuning_freq0, i0p)
                  + H_selfkerr(xi0, i0p);
          }
          double yre = (-hd + hdp ) * xim;
          double yim = ( hd - hdp ) * xre;

          // Decay l1^T, diagonal part: xout += l1diag xin
          // Dephasing l2^T: xout += l2(ik, ikp) xin
          if (shellctx->lindbladtype != LindbladType::NONE) {
            double l1diag = L1diag(decay0, i0, i0p);
            double l2 = L2(dephase0, i0, i0p);
            yre += (l2 + l1diag) * xre;
            yim += (l2 + l1diag) * xim;
          }

          /* --- Offdiagonal coupling term J_kl --- */
 
          /* --- Offdiagonal part of decay L1^T */
          if (shellctx->lindbladtype != LindbladType::NONE) {
            // Oscillators 0
            L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
          }

          /* --- Control hamiltonian  --- */
          // Oscillator 0
          control_T(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);

          /* Update */
          yptr[2*it]   = yre;
          yptr[2*it+1] = yim;
          it++;
      }
  }

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms_diff(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }



  /* Restore x and y */
  VecRestoreArrayRead(x, &xptr);
  VecRestoreArray(y, &yptr);

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double cos01 = cos(eta01 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
  int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
  int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
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
          double hdp = 0.0;
          if (shellctx->lindbladtype != LindbladType::NONE) {
            hdp = H_detune(detuning_freq0, detuning_freq1, i0p, i1p)
                + H_selfkerr(xi0, xi1, i0p, i1p)
                + H_crosskerr(xi01, i0p, i1p);
          }
          double yre = ( hd - hdp ) * xim;
          double yim = (-hd + hdp ) * xre;

          // Decay l1, diagonal part: xout += l1diag xin
          // Dephasing l2: xout += l2(ik, ikp) xin
          if (shellctx->lindbladtype != LindbladType::NONE) {
            double l1diag = L1diag(decay0, decay1, i0, i1, i0p, i1p);
            double l2 = L2(dephase0, dephase1, i0, i1, i0p, i1p);
            yre += (l2 + l1diag) * xre;
            yim += (l2 + l1diag) * xim;
          }

          /* --- Offdiagonal: Jkl coupling term --- */
          // oscillator 0<->1 
          Jkl_coupling(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);

          /* --- Offdiagonal part of decay L1 */
          if (shellctx->lindbladtype != LindbladType::NONE) {
            // Oscillators 0
            L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
            // Oscillator 1
            L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
          }

          /* --- Control hamiltonian --- */
          // Oscillator 0 
          control(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
          // Oscillator 1
          control(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);

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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double cos01 = cos(eta01 * shellctx->time);
  double sin01 = sin(eta01 * shellctx->time);

  /* compute strides for accessing x at i0+1, i0-1, i0p+1, i0p-1, i1+1, i1-1, i1p+1, i1p-1: */
  int stridei0  = TensorGetIndex(n0,n1, 1,0,0,0);
  int stridei1  = TensorGetIndex(n0,n1, 0,1,0,0);
  int stridei0p = TensorGetIndex(n0,n1, 0,0,1,0);
  int stridei1p = TensorGetIndex(n0,n1, 0,0,0,1);

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
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
          double hdp = 0.0;
          if (shellctx->lindbladtype != LindbladType::NONE) {
            hdp = H_detune(detuning_freq0, detuning_freq1, i0p, i1p)
                  + H_selfkerr(xi0, xi1, i0p, i1p)
                  + H_crosskerr(xi01, i0p, i1p);
          }
          double yre = (-hd + hdp ) * xim;
          double yim = ( hd - hdp ) * xre;

          // Decay l1^T, diagonal part: xout += l1diag xin
          // Dephasing l2^T: xout += l2(ik, ikp) xin
          if (shellctx->lindbladtype != LindbladType::NONE) {
            double l1diag = L1diag(decay0, decay1, i0, i1, i0p, i1p);
            double l2 = L2(dephase0, dephase1, i0, i1, i0p, i1p);
            yre += (l2 + l1diag) * xre;
            yim += (l2 + l1diag) * xim;
          }

          /* --- Offdiagonal coupling term J_kl --- */
          // oscillator 0<->1
          Jkl_coupling_T(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
 
          /* --- Offdiagonal part of decay L1^T */
          if (shellctx->lindbladtype != LindbladType::NONE) {
            // Oscillators 0
            L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
            // Oscillator 1
            L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
          }

          /* --- Control hamiltonian  --- */
          // Oscillator 0
          control_T(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
          // Oscillator 1
          control_T(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);

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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms_diff(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double pt2 = shellctx->control_Re[2][0];
  double qt2 = shellctx->control_Im[2][0];
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

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  int n2p = n2;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
    n2p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
      for (int i2p = 0; i2p < n2p; i2p++)  {
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
              double hdp =0.0;
              if (shellctx->lindbladtype != LindbladType::NONE) {
                hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, i0p, i1p, i2p)
                      + H_selfkerr(xi0, xi1, xi2, i0p, i1p, i2p)
                      + H_crosskerr(xi01, xi02, xi12, i0p, i1p, i2p);
              }
              double yre = ( hd - hdp ) * xim;
              double yim = (-hd + hdp ) * xre;

              // Decay l1, diagonal part: xout += l1diag xin
              // Dephasing l2: xout += l2(ik, ikp) xin
              if (shellctx->lindbladtype != LindbladType::NONE) {
                double l1diag = L1diag(decay0, decay1, decay2, i0, i1, i2, i0p, i1p, i2p);
                double l2 = L2(dephase0, dephase1, dephase2, i0, i1, i2, i0p, i1p, i2p);
                yre += (l2 + l1diag) * xre;
                yim += (l2 + l1diag) * xim;
              }

              /* --- Offdiagonal: Jkl coupling  --- */
              // oscillator 0<->1 
              Jkl_coupling(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
              // oscillator 0<->2
              Jkl_coupling(it, n0, n2, n0p, n2p, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
              // oscillator 1<->2
              Jkl_coupling(it, n1, n2, n1p, n2p, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);

              /* --- Offdiagonal part of decay L1 */
              if (shellctx->lindbladtype != LindbladType::NONE) {
                // Oscillators 0
                L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                // Oscillator 1
                L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                // Oscillator 2
                L1decay(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
              }

              /* --- Control hamiltonian ---  */
              // Oscillator 0 
              control(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
              // Oscillator 1
              control(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
              // Oscillator 1
              control(it, n2, i2, n2p, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
              
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
  
  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double pt2 = shellctx->control_Re[2][0];
  double qt2 = shellctx->control_Im[2][0];
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

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  int n2p = n2;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
    n2p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
      for (int i2p = 0; i2p < n2p; i2p++)  {
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
              double hdp = 0.0;
              if (shellctx->lindbladtype != LindbladType::NONE) {
                hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, i0p, i1p, i2p)
                    + H_selfkerr(xi0, xi1, xi2, i0p, i1p, i2p)
                    + H_crosskerr(xi01, xi02, xi12, i0p, i1p, i2p);
              }
              double yre = (-hd + hdp ) * xim;
              double yim = ( hd - hdp ) * xre;

              // Decay l1^T, diagonal part: xout += l1diag xin
              // Dephasing l2^T: xout += l2(ik, ikp) xin
              if (shellctx->lindbladtype != LindbladType::NONE) {
                double l1diag = L1diag(decay0, decay1, decay2, i0, i1, i2, i0p, i1p, i2p);
                double l2 = L2(dephase0, dephase1, dephase2, i0, i1, i2, i0p, i1p, i2p);
                yre += (l2 + l1diag) * xre;
                yim += (l2 + l1diag) * xim;
              }

              /* --- Offdiagonal coupling term J_kl --- */
              // oscillator 0<->1
              Jkl_coupling_T(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
              // oscillator 0<->2
              Jkl_coupling_T(it, n0, n2, n0p, n2p, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
              // oscillator 1<->2
              Jkl_coupling_T(it, n1, n2, n1p, n2p, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
              

              /* --- Offdiagonal part of decay L1^T */
              if (shellctx->lindbladtype != LindbladType::NONE) {
                // Oscillators 0
                L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                // Oscillator 1
                L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                // Oscillator 2
                L1decay_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
              }

              /* --- Control hamiltonian  --- */
              // Oscillator 0
              control_T(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
              // Oscillator 1
              control_T(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
              // Oscillator 2
              control_T(it, n2, i2, n2p, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);

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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms_diff(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }
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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double pt2 = shellctx->control_Re[2][0];
  double qt2 = shellctx->control_Im[2][0];
  double pt3 = shellctx->control_Re[3][0];
  double qt3 = shellctx->control_Im[3][0];
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

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  int n2p = n2;
  int n3p = n3;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
    n2p = 1;
    n3p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
      for (int i2p = 0; i2p < n2p; i2p++)  {
        for (int i3p = 0; i3p < n3p; i3p++)  {
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
                  double hdp = 0.0;
                  if (shellctx->lindbladtype != LindbladType::NONE) {
                    hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, i0p, i1p, i2p, i3p)
                          + H_selfkerr(xi0, xi1, xi2, xi3, i0p, i1p, i2p, i3p)
                          + H_crosskerr(xi01, xi02, xi03, xi12, xi13, xi23, i0p, i1p, i2p, i3p);
                  }
                  double yre = ( hd - hdp ) * xim;
                  double yim = (-hd + hdp ) * xre;

                  if (shellctx->lindbladtype != LindbladType::NONE) {
                    // Decay l1, diagonal part: xout += l1diag xin
                    // Dephasing l2: xout += l2(ik, ikp) xin
                    double l1diag = L1diag(decay0, decay1, decay2, decay3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                    double l2 = L2(dephase0, dephase1, dephase2, dephase3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                    yre += (l2 + l1diag) * xre;
                    yim += (l2 + l1diag) * xim;
                  }

                  /* --- Offdiagonal: Jkl coupling  --- */
                  // oscillator 0<->1 
                  Jkl_coupling(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                  // oscillator 0<->2
                  Jkl_coupling(it, n0, n2, n0p, n2p, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                  // oscillator 0<->3
                  Jkl_coupling(it, n0, n3, n0p, n3p, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                  // oscillator 1<->2
                  Jkl_coupling(it, n1, n2, n1p, n2p, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                  // oscillator 1<->3
                  Jkl_coupling(it, n1, n3, n1p, n3p, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                  // oscillator 2<->3
                  Jkl_coupling(it, n2, n3, n2p, n3p, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);

                  /* --- Offdiagonal part of decay L1 */
                  if (shellctx->lindbladtype != LindbladType::NONE) {
                    // Oscillators 0
                    L1decay(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                    // Oscillator 1
                    L1decay(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                    // Oscillator 2
                    L1decay(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
                    // Oscillator 3
                    L1decay(it, n3, i3, i3p, stridei3, stridei3p, xptr, decay3, &yre, &yim);
                  }
              
                  /* --- Control hamiltonian ---  */
                  // Oscillator 0 
                  control(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                  // Oscillator 1
                  control(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                  // Oscillator 2
                  control(it, n2, i2, n2p, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                  // Oscillator 2
                  control(it, n3, i3, n3p, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);
              
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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double pt2 = shellctx->control_Re[2][0];
  double qt2 = shellctx->control_Im[2][0];
  double pt3 = shellctx->control_Re[3][0];
  double qt3 = shellctx->control_Im[3][0];
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

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  int n2p = n2;
  int n3p = n3;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
    n2p = 1;
    n3p = 1;
  }


   /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
      for (int i2p = 0; i2p < n2p; i2p++)  {
        for (int i3p = 0; i3p < n3p; i3p++)  {
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
                  double hdp = 0.0;
                  if (shellctx->lindbladtype != LindbladType::NONE) {
                    hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, i0p, i1p, i2p, i3p)
                             + H_selfkerr(xi0, xi1, xi2, xi3, i0p, i1p, i2p, i3p)
                             + H_crosskerr(xi01, xi02, xi03, xi12, xi13, xi23, i0p, i1p, i2p, i3p);
                  }
                  double yre = (-hd + hdp ) * xim;
                  double yim = ( hd - hdp ) * xre;

                  // Decay l1^T, diagonal part: xout += l1diag xin
                  // Dephasing l2^T: xout += l2(ik, ikp) xin
                  if (shellctx->lindbladtype != LindbladType::NONE) {
                    double l1diag = L1diag(decay0, decay1, decay2, decay3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                    double l2 = L2(dephase0, dephase1, dephase2, dephase3, i0, i1, i2, i3, i0p, i1p, i2p, i3p);
                    yre += (l2 + l1diag) * xre;
                    yim += (l2 + l1diag) * xim;
                  }

                  /* --- Offdiagonal coupling term J_kl --- */
                  // oscillator 0<->1
                  Jkl_coupling_T(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                  // oscillator 0<->2
                  Jkl_coupling_T(it, n0, n2, n0p, n2p, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                  // oscillator 0<->3
                  Jkl_coupling_T(it, n0, n3, n0p, n3p, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                  // oscillator 1<->2
                  Jkl_coupling_T(it, n1, n2, n1p, n2p, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                  // oscillator 1<->3
                  Jkl_coupling_T(it, n1, n3, n1p, n3p, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                  // oscillator 2<->3
                  Jkl_coupling_T(it, n2, n3, n2p, n3p, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);
              

                  /* --- Offdiagonal part of decay L1^T */
                  if (shellctx->lindbladtype != LindbladType::NONE) {
                    // Oscillators 0
                    L1decay_T(it, n0, i0, i0p, stridei0, stridei0p, xptr, decay0, &yre, &yim);
                    // Oscillator 1
                    L1decay_T(it, n1, i1, i1p, stridei1, stridei1p, xptr, decay1, &yre, &yim);
                    // Oscillator 2
                    L1decay_T(it, n2, i2, i2p, stridei2, stridei2p, xptr, decay2, &yre, &yim);
                    // Oscillator 3
                    L1decay_T(it, n3, i3, i3p, stridei3, stridei3p, xptr, decay3, &yre, &yim);
                  }

                  /* --- Control hamiltonian  --- */
                  // Oscillator 0
                  control_T(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                  // Oscillator 1
                  control_T(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                  // Oscillator 2
                  control_T(it, n2, i2, n2p, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                  // Oscillator 3
                  control_T(it, n3, i3, n3p, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);

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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms_diff(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double pt2 = shellctx->control_Re[2][0];
  double qt2 = shellctx->control_Im[2][0];
  double pt3 = shellctx->control_Re[3][0];
  double qt3 = shellctx->control_Im[3][0];
  double pt4 = shellctx->control_Re[4][0];
  double qt4 = shellctx->control_Im[4][0];
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

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  int n2p = n2;
  int n3p = n3;
  int n4p = n4;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
    n2p = 1;
    n3p = 1;
    n4p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
      for (int i2p = 0; i2p < n2p; i2p++)  {
        for (int i3p = 0; i3p < n3p; i3p++)  {
          for (int i4p = 0; i4p < n4p; i4p++)  {
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
                      double hdp = 0.0;
                      if (shellctx->lindbladtype != LindbladType::NONE) {
                        hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, detuning_freq4, i0p, i1p, i2p, i3p, i4p)
                                 + H_selfkerr(xi0, xi1, xi2, xi3, xi4, i0p, i1p, i2p, i3p, i4p)
                                 + H_crosskerr(xi01, xi02, xi03, xi04, xi12, xi13, xi14, xi23, xi24, xi34, i0p, i1p, i2p, i3p, i4p);
                      }
                      double yre = ( hd - hdp ) * xim;
                      double yim = (-hd + hdp ) * xre;

                      if (shellctx->lindbladtype != LindbladType::NONE) {
                        // Decay l1, diagonal part: xout += l1diag xin
                        // Dephasing l2: xout += l2(ik, ikp) xin
                        double l1diag = L1diag(decay0, decay1, decay2, decay3, decay4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                        double l2 = L2(dephase0, dephase1, dephase2, dephase3, dephase4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                        yre += (l2 + l1diag) * xre;
                        yim += (l2 + l1diag) * xim;
                      }

                      /* --- Offdiagonal: Jkl coupling  --- */
                      // oscillator 0<->1 
                      Jkl_coupling(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                      // oscillator 0<->2
                      Jkl_coupling(it, n0, n2, n0p, n2p, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                      // oscillator 0<->3
                      Jkl_coupling(it, n0, n3, n0p, n3p, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                      // oscillator 0<->4
                      Jkl_coupling(it, n0, n4, n0p, n4p, i0, i0p, i4, i4p, stridei0, stridei0p, stridei4, stridei4p, xptr, J04, cos04, sin04, &yre, &yim);
                      // oscillator 1<->2
                      Jkl_coupling(it, n1, n2, n1p, n2p, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                      // oscillator 1<->3
                      Jkl_coupling(it, n1, n3, n1p, n3p, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                      // oscillator 1<->4
                      Jkl_coupling(it, n1, n4, n1p, n4p, i1, i1p, i4, i4p, stridei1, stridei1p, stridei4, stridei4p, xptr, J14, cos14, sin14, &yre, &yim);
                      // oscillator 2<->3
                      Jkl_coupling(it, n2, n3, n2p, n3p, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);
                      // oscillator 2<->4
                      Jkl_coupling(it, n2, n4, n2p, n4p, i2, i2p, i4, i4p, stridei2, stridei2p, stridei4, stridei4p, xptr, J24, cos24, sin24, &yre, &yim);
                      // oscillator 3<->4
                      Jkl_coupling(it, n3, n4, n3p, n4p, i3, i3p, i4, i4p, stridei3, stridei3p, stridei4, stridei4p, xptr, J34, cos34, sin34, &yre, &yim);

                      /* --- Offdiagonal part of decay L1 */
                      if (shellctx->lindbladtype != LindbladType::NONE) {
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
                      }

                      /* --- Control hamiltonian ---  */
                      // Oscillator 0 
                      control(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                      // Oscillator 1
                      control(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                      // Oscillator 2
                      control(it, n2, i2, n2p, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                      // Oscillator 3
                      control(it, n3, i3, n3p, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);
                      // Oscillator 4
                      control(it, n4, i4, n4p, i4p, stridei4, stridei4p, xptr, pt4, qt4, &yre, &yim);
              
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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }

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
  double pt0 = shellctx->control_Re[0][0];
  double qt0 = shellctx->control_Im[0][0];
  double pt1 = shellctx->control_Re[1][0];
  double qt1 = shellctx->control_Im[1][0];
  double pt2 = shellctx->control_Re[2][0];
  double qt2 = shellctx->control_Im[2][0];
  double pt3 = shellctx->control_Re[3][0];
  double qt3 = shellctx->control_Im[3][0];
  double pt4 = shellctx->control_Re[4][0];
  double qt4 = shellctx->control_Im[4][0];
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

  /* Switch for Lindblad vs Schroedinger solver */
  int n0p = n0;
  int n1p = n1;
  int n2p = n2;
  int n3p = n3;
  int n4p = n4;
  if (shellctx->lindbladtype == LindbladType::NONE) { // Schroedinger
    n0p = 1; // Cut down so that below loop has i0p=0 and i1p=0/
    n1p = 1;
    n2p = 1;
    n3p = 1;
    n4p = 1;
  }

  /* Iterate over indices of output vector y */
  int it = 0;
  for (int i0p = 0; i0p < n0p; i0p++)  {
    for (int i1p = 0; i1p < n1p; i1p++)  {
      for (int i2p = 0; i2p < n2p; i2p++)  {
        for (int i3p = 0; i3p < n3p; i3p++)  {
          for (int i4p = 0; i4p < n4p; i4p++)  {
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
                      double hdp = 0.0;
                      if (shellctx->lindbladtype != LindbladType::NONE) { 
                        hdp = H_detune(detuning_freq0, detuning_freq1, detuning_freq2, detuning_freq3, detuning_freq4, i0p, i1p, i2p, i3p, i4p)
                                 + H_selfkerr(xi0, xi1, xi2, xi3, xi4, i0p, i1p, i2p, i3p, i4p)
                                 + H_crosskerr(xi01, xi02, xi03, xi04, xi12, xi13, xi14, xi23, xi24, xi34, i0p, i1p, i2p, i3p, i4p);
                      }
                      double yre = (-hd + hdp ) * xim;
                      double yim = ( hd - hdp ) * xre;

                      // Decay l1^T, diagonal part: xout += l1diag xin
                      // Dephasing l2^T: xout += l2(ik, ikp) xin
                      if (shellctx->lindbladtype != LindbladType::NONE) {
                        double l1diag = L1diag(decay0, decay1, decay2, decay3, decay4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                        double l2 = L2(dephase0, dephase1, dephase2, dephase3, dephase4, i0, i1, i2, i3, i4, i0p, i1p, i2p, i3p, i4p);
                        yre += (l2 + l1diag) * xre;
                        yim += (l2 + l1diag) * xim;
                      }

                      /* --- Offdiagonal coupling term J_kl --- */
                      // oscillator 0<->1
                      Jkl_coupling_T(it, n0, n1, n0p, n1p, i0, i0p, i1, i1p, stridei0, stridei0p, stridei1, stridei1p, xptr, J01, cos01, sin01, &yre, &yim);
                      // oscillator 0<->2
                      Jkl_coupling_T(it, n0, n2, n0p, n2p, i0, i0p, i2, i2p, stridei0, stridei0p, stridei2, stridei2p, xptr, J02, cos02, sin02, &yre, &yim);
                      // oscillator 0<->3
                      Jkl_coupling_T(it, n0, n3, n0p, n3p, i0, i0p, i3, i3p, stridei0, stridei0p, stridei3, stridei3p, xptr, J03, cos03, sin03, &yre, &yim);
                      // oscillator 0<->4
                      Jkl_coupling_T(it, n0, n4, n0p, n4p, i0, i0p, i4, i4p, stridei0, stridei0p, stridei4, stridei4p, xptr, J04, cos04, sin04, &yre, &yim);
                      // oscillator 1<->2
                      Jkl_coupling_T(it, n1, n2, n1p, n2p, i1, i1p, i2, i2p, stridei1, stridei1p, stridei2, stridei2p, xptr, J12, cos12, sin12, &yre, &yim);
                      // oscillator 1<->3
                      Jkl_coupling_T(it, n1, n3, n1p, n3p, i1, i1p, i3, i3p, stridei1, stridei1p, stridei3, stridei3p, xptr, J13, cos13, sin13, &yre, &yim);
                      // oscillator 1<->4
                      Jkl_coupling_T(it, n1, n4, n1p, n4p, i1, i1p, i4, i4p, stridei1, stridei1p, stridei4, stridei4p, xptr, J14, cos14, sin14, &yre, &yim);
                      // oscillator 2<->3
                      Jkl_coupling_T(it, n2, n3, n2p, n3p, i2, i2p, i3, i3p, stridei2, stridei2p, stridei3, stridei3p, xptr, J23, cos23, sin23, &yre, &yim);
                      // oscillator 2<->4
                      Jkl_coupling_T(it, n2, n4, n2p, n4p, i2, i2p, i4, i4p, stridei2, stridei2p, stridei4, stridei4p, xptr, J24, cos24, sin24, &yre, &yim);
                      // oscillator 3<->4
                      Jkl_coupling_T(it, n3, n4, n3p, n4p, i3, i3p, i4, i4p, stridei3, stridei3p, stridei4, stridei4p, xptr, J34, cos34, sin34, &yre, &yim);
              
                      /* --- Offdiagonal part of decay L1^T */
                      if (shellctx->lindbladtype != LindbladType::NONE) { 
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
                      }

                      /* --- Control hamiltonian  --- */
                      // Oscillator 0
                      control_T(it, n0, i0, n0p, i0p, stridei0, stridei0p, xptr, pt0, qt0, &yre, &yim);
                      // Oscillator 1
                      control_T(it, n1, i1, n1p, i1p, stridei1, stridei1p, xptr, pt1, qt1, &yre, &yim);
                      // Oscillator 2
                      control_T(it, n2, i2, n2p, i2p, stridei2, stridei2p, xptr, pt2, qt2, &yre, &yim);
                      // Oscillator 3
                      control_T(it, n3, i3, n3p, i3p, stridei3, stridei3p, xptr, pt3, qt3, &yre, &yim);
                      // Oscillator 4
                      control_T(it, n4, i4, n4p, i4p, stridei4, stridei4p, xptr, pt4, qt4, &yre, &yim);

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

  /* Apply learning */
  if (shellctx->useUDEmodel) {
    Vec u, v, uout, vout;
    VecGetSubVector(x, *shellctx->isu, &u);
    VecGetSubVector(x, *shellctx->isv, &v);
    VecGetSubVector(y, *shellctx->isu, &uout);
    VecGetSubVector(y, *shellctx->isv, &vout);
    shellctx->learning->applyLearningTerms_diff(u,v,uout, vout);
    VecRestoreSubVector(x, *shellctx->isu, &u);
    VecRestoreSubVector(x, *shellctx->isv, &v);
    VecRestoreSubVector(y, *shellctx->isu, &uout);
    VecRestoreSubVector(y, *shellctx->isv, &vout);
  }
  return 0;
}


/* --- 1 Oscillator cases --- */
int myMatMult_matfree_1Osc(Mat RHS, Vec x, Vec y){
  /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  if      (n0==2)  return myMatMult_matfree<2>(RHS, x, y);
  else if (n0==3)  return myMatMult_matfree<3>(RHS, x, y);
  else if (n0==4)  return myMatMult_matfree<4>(RHS, x, y);
  else if (n0==5)  return myMatMult_matfree<5>(RHS, x, y);
  else if (n0==6)  return myMatMult_matfree<6>(RHS, x, y);
  else if (n0==7)  return myMatMult_matfree<7>(RHS, x, y);
  else if (n0==8)  return myMatMult_matfree<8>(RHS, x, y);
  else if (n0==9)  return myMatMult_matfree<9>(RHS, x, y);
  else if (n0==10)  return myMatMult_matfree<10>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("\nERROR: Matrix-free solver for this number of qubit levels needs a simple modification:\n");
      printf("  Add the following lines to the end of src/mastereq.cpp and recompile Quandary:\n \
  -> In function 'int myMatMult_matfree_1Osc(..)':  \n \
             elseif (n0==%d) return  myMatMult_matfree<%d>(RHS, x, y); \n \
  -> In function 'int myMatMultTranspose_matfree_1Osc(..)': \n \
             elseif (n0==%d) return  myMatMultTranspose_matfree<%d>(RHS, x, y);\n\n", n0, n0, n0, n0);
      exit(1);
    } 
    return 0;
  }
}
int myMatMultTranspose_matfree_1Osc(Mat RHS, Vec x, Vec y){
 /* Get the shell context */
  MatShellCtx *shellctx;
  MatShellGetContext(RHS, (void**) &shellctx);
  int n0 = shellctx->nlevels[0];
  if      (n0==2)  return myMatMultTranspose_matfree<2>(RHS, x, y);
  else if (n0==3)  return myMatMultTranspose_matfree<3>(RHS, x, y);
  else if (n0==4)  return myMatMultTranspose_matfree<4>(RHS, x, y);
  else if (n0==5)  return myMatMultTranspose_matfree<5>(RHS, x, y);
  else if (n0==6)  return myMatMultTranspose_matfree<6>(RHS, x, y);
  else if (n0==7)  return myMatMultTranspose_matfree<7>(RHS, x, y);
  else if (n0==8)  return myMatMultTranspose_matfree<8>(RHS, x, y);
  else if (n0==9)  return myMatMultTranspose_matfree<9>(RHS, x, y);
  else if (n0==10)  return myMatMultTranspose_matfree<10>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("ERROR: Matrix-free solver for this number of qubit levels needs template instantiation.\n");
      exit(1);
    }
    return 0;
  }
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
  else if (n0==1 && n1==1)   return myMatMult_matfree<1,1>(RHS, x, y);
  else if (n0==2 && n1==2)   return myMatMult_matfree<2,2>(RHS, x, y);
  else if (n0==3 && n1==3)   return myMatMult_matfree<3,3>(RHS, x, y);
  else if (n0==4 && n1==4)   return myMatMult_matfree<4,4>(RHS, x, y);
  else if (n0==5 && n1==5)   return myMatMult_matfree<5,5>(RHS, x, y);
  else if (n0==10 && n1==10)   return myMatMult_matfree<10,10>(RHS, x, y);
  else if (n0==20 && n1==20) return myMatMult_matfree<20,20>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("\nERROR: Matrix-free solver for this number of qubit levels needs a simple modification:\n");
      printf("  Add the following lines to the end of src/mastereq.cpp and recompile Quandary:\n \
  -> In function 'int myMatMult_matfree_2Osc(..)':  \n \
             elseif (n0==%d && n1==%d) return  myMatMult_matfree<%d,%d>(RHS, x, y); \n \
  -> In function 'int myMatMultTranspose_matfree_2Osc(..)': \n \
             elseif (n0==%d && n1==%d) return  myMatMultTranspose_matfree<%d,%d>(RHS, x, y);\n\n", n0, n1, n0, n1, n0, n1, n0, n1);
      exit(1);
    }
    return 0;
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
  else if (n0==1 && n1==1)   return myMatMultTranspose_matfree<1,1>(RHS, x, y);
  else if (n0==2 && n1==2)   return myMatMultTranspose_matfree<2,2>(RHS, x, y);
  else if (n0==3 && n1==3)   return myMatMultTranspose_matfree<3,3>(RHS, x, y);
  else if (n0==4 && n1==4)   return myMatMultTranspose_matfree<4,4>(RHS, x, y);
  else if (n0==5 && n1==5)   return myMatMultTranspose_matfree<5,5>(RHS, x, y);
  else if (n0==10 && n1==10)   return myMatMultTranspose_matfree<10,10>(RHS, x, y);
  else if (n0==20 && n1==20) return myMatMultTranspose_matfree<20,20>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("ERROR: Matrix-free solver for this number of qubit levels needs template instanciation.\n");
      exit(1);
    }
    return 0;
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
  else if (n0==4 && n1==4 && n2==4) return myMatMult_matfree<4,4,4>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("\nERROR: Matrix-free solver for this number of qubit levels needs a simple modification:\n");
      printf("  Add the following lines to the end of src/mastereq.cpp and recompile Quandary:\n \
  -> In function 'int myMatMult_matfree_3Osc(..)':  \n \
             elseif (n0==%d && n1==%d && n2==%d) return  myMatMult_matfree<%d,%d,%d>(RHS, x, y); \n \
  -> In function 'int myMatMultTranspose_matfree_3Osc(..)': \n \
             elseif (n0==%d && n1==%d && n2==%d) return  myMatMultTranspose_matfree<%d,%d,%d>(RHS, x, y);\n\n", n0, n1, n2, n0, n1, n2, n0, n1, n2, n0, n1, n2);
      exit(1);
    } 
    return 0;
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
  else if (n0==4 && n1==4 && n2==4)  return myMatMultTranspose_matfree<4,4,4>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("ERROR: Matrix-free solver for this number of qubit levels needs template instanciation.\n");
      exit(1);
    }
    return 0;
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
  else if (n0==3 && n1==3 && n2==3 && n3 == 3) return myMatMult_matfree<3,3,3,3>(RHS, x, y);
  else if (n0==4 && n1==4 && n2==4 && n3 == 4) return myMatMult_matfree<4,4,4,4>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("\nERROR: Matrix-free solver for this number of qubit levels needs a simple modification:\n");
      printf("  Add the following lines to the end of src/mastereq.cpp and recompile Quandary:\n \
  -> In function 'int myMatMult_matfree_4Osc(..)':  \n \
             elseif (n0==%d && n1==%d && n2==%d && n3==%d) return  myMatMult_matfree<%d,%d,%d,%d>(RHS, x, y); \n \
  -> In function 'int myMatMultTranspose_matfree_4Osc(..)': \n \
             elseif (n0==%d && n1==%d && n2==%d && n3==%d) return  myMatMultTranspose_matfree<%d,%d,%d,%d>(RHS, x, y);\n\n", n0, n1, n2, n3, n0, n1, n2, n3, n0, n1, n2, n3, n0, n1, n2, n3);
      exit(1);
    }
    return 0;
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
  else if (n0==3 && n1==3 && n2==3 && n3==3)  return myMatMultTranspose_matfree<3,3,3,3>(RHS, x, y);
  else if (n0==4 && n1==4 && n2==4 && n3==4)  return myMatMultTranspose_matfree<4,4,4,4>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("ERROR: Matrix-free solver for this number of qubit levels needs template instanciation.\n");
      exit(1);
    }
    return 0;
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
  else if (n0==3 && n1==3 && n2==3 && n3 == 3 && n4 == 3) return myMatMult_matfree<3,3,3,3,3>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("\nERROR: Matrix-free solver for this number of qubit levels needs a simple modification:\n");
      printf("  Add the following lines to the end of src/mastereq.cpp and recompile Quandary:\n \
  -> In function 'int myMatMult_matfree_5Osc(..)':  \n \
             elseif (n0==%d && n1==%d && n2==%d && n3==%d && n4==%d) return  myMatMult_matfree<%d,%d,%d,%d,%d>(RHS, x, y); \n \
  -> In function 'int myMatMultTranspose_matfree_5Osc(..)': \n \
             elseif (n0==%d && n1==%d && n2==%d && n3==%d && n4==%d) return  myMatMultTranspose_matfree<%d,%d,%d,%d,%d>(RHS, x, y);\n\n", n0, n1, n2, n3, n4, n0, n1, n2, n3, n4, n0, n1, n2, n3, n4, n0, n1, n2, n3, n4);
      exit(1);
    }
    return 0;
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
  else if (n0==3 && n1==3 && n2==3 && n3==3 && n4==3)  return myMatMultTranspose_matfree<3,3,3,3,3>(RHS, x, y);
  else {
    int mpirank_world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    if (mpirank_world==0) {
      printf("ERROR: Matrix-free solver for this number of qubit levels needs template instanciation.\n");
      exit(1);
    }
    return 0;
  }
}