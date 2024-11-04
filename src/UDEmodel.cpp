#include "UDEmodel.hpp"

UDEmodel::UDEmodel(){
  dim_rho = 0;
  dim = 0;
  nparams = 0;
}

UDEmodel::UDEmodel(int dim_rho_, LindbladType lindblad_type){
  dim_rho = dim_rho_;
  // If Lindblad solver, dim = N^2, otherwise dim=N
  dim = dim_rho;
  if (lindblad_type != LindbladType::NONE){
    dim = dim_rho*dim_rho; 
  }
  
  /* Allocate an auxiliary vector for system matrix matmult */
  VecCreate(PETSC_COMM_WORLD, &aux);     // aux sized for Re(state) or Im(state) 
  VecSetSizes(aux , PETSC_DECIDE, dim);
  VecSetFromOptions(aux);
}

UDEmodel::~UDEmodel(){
  for (int i=0; i<SystemMats_A.size(); i++) MatDestroy(&SystemMats_A[i]);
  for (int i=0; i<SystemMats_B.size(); i++) MatDestroy(&SystemMats_B[i]);
  SystemMats_A.clear();
  SystemMats_B.clear();
  VecDestroy(&aux);
  for (int i=0; i<Operator.size(); i++) MatDestroy(&Operator[i]);
}

HamiltonianModel::HamiltonianModel(int dim_rho_, bool shifted_diag_, LindbladType lindbladtype) : UDEmodel(dim_rho_, lindbladtype) {
  shifted_diag = shifted_diag_;

  /* Assemble system Matrices */
  createSystemMats(lindbladtype);

  /* Set the number of learnable parameters */
  nparams = SystemMats_A.size() + SystemMats_B.size();

  /* Allocate storage for returning learned Hamitonian operators */
  Operator.resize(2);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator[0]);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator[1]);
  MatSetUp(Operator[0]);
  MatSetUp(Operator[1]);
}

HamiltonianModel::~HamiltonianModel(){}

void HamiltonianModel::createSystemMats(LindbladType lindbladtype){

  /* Set up and store the Hamiltonian system matrices:
   *   (-i*sigma)   or vectorized   -i(I kron sigma - sigma^T kron I) 
   *  A = Re(-isigma)
   *  B = Im(-isigma)
   */

  /* Create the Gellmann matrices*/
  std::vector<Mat> BasisMats_Re, BasisMats_Im;
  createGellmannMats(dim_rho, false, false, shifted_diag, true, BasisMats_Re, BasisMats_Im);
  // Note BasisMats[0] contains the identity. Grab it here:
  Mat Id = BasisMats_Re[0];

  // Set up -i*(Real_Gellmann), they go into system mat Bd = Im(-iH)
  for (int i=1; i<BasisMats_Re.size(); i++){
    Mat myMat;
    if (lindbladtype == LindbladType::NONE){ // -sigma
      MatDuplicate(BasisMats_Re[i],  MAT_COPY_VALUES, &myMat);
      MatScale(myMat, -1.0);
    } else { // - I kron sigma + sigma^T kron I
      Mat myMat1, myMat2;
      MatTranspose(BasisMats_Re[i], MAT_INITIAL_MATRIX, &myMat1);      // myMat1 = sigma^T
      MatSeqAIJKron(myMat1, Id, MAT_INITIAL_MATRIX, &myMat);          // myMat = sigma^T kron I
      MatSeqAIJKron(Id, BasisMats_Re[i], MAT_INITIAL_MATRIX, &myMat2); // myMat2 = I kron sigma
      MatAXPY(myMat, -1.0, myMat2, DIFFERENT_NONZERO_PATTERN);        // myMat = sigma^T kron I - I kron sigma
      MatDestroy(&myMat1);
      MatDestroy(&myMat2);
    }
    SystemMats_B.push_back(myMat);
  }

  // Set up -i*(Imag_BasisMat), they go into system mat Ad = Re(-iH) 
  Mat myMat;
  for (int i=0; i<BasisMats_Im.size(); i++){
    if (lindbladtype == LindbladType::NONE){ // sigma
      MatDuplicate(BasisMats_Im[i],  MAT_COPY_VALUES, &myMat);
    } else { // I kron sigma - sigma^T kron I
      Mat myMat1, myMat2;
      MatSeqAIJKron(Id, BasisMats_Im[i], MAT_INITIAL_MATRIX, &myMat); // myMat = I kron sigma
      MatTranspose(BasisMats_Im[i], MAT_INITIAL_MATRIX, &myMat1);    // myMat1 = sigma^T
      MatSeqAIJKron(myMat1, Id, MAT_INITIAL_MATRIX, &myMat2);       // myMat2 = sigma^T kron I
      MatAXPY(myMat, -1.0, myMat2, DIFFERENT_NONZERO_PATTERN);      // myMat = I kron sigma - sigma^T kron I
      MatDestroy(&myMat1);
      MatDestroy(&myMat2);
    }
    SystemMats_A.push_back(myMat);
  }

  /* Destroy the basis matrices */
  for (int i=0; i<BasisMats_Re.size(); i++) MatDestroy(&BasisMats_Re[i]);
  for (int i=0; i<BasisMats_Im.size(); i++) MatDestroy(&BasisMats_Im[i]);
  BasisMats_Re.clear();
  BasisMats_Im.clear();
}


void HamiltonianModel::applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsH){
// Note: All purely real Basis Mats correspond to purely imaginary System matrices (stored in in SystemMatsB). Hence, the learnparams_Re (which go for BasisMats_Re), are applied to SystemMats_B, and learnparams_Im are applied to SystemMats_A!

  /* Extract pointers to params that correspond to SystemMats_A vs _B */
  // Assume learnparams = [learnparams_re, learnparams_Im]
  double* paramsB = learnparamsH.data();                      // learnparams_re -> SystemMatsB
  double* paramsA = learnparamsH.data()+ SystemMats_B.size(); // learnparams_im -> SystemMatsA

  // Real parts of (-i * H)
  for (int i=0; i< SystemMats_A.size(); i++){
    // uout += learnparam_Im * SystemA * u
    MatMult(SystemMats_A[i], u, aux);
    VecAXPY(uout, paramsA[i], aux); 
    // vout += learnparam_IM * SystemA * v
    MatMult(SystemMats_A[i], v, aux);
    VecAXPY(vout, paramsA[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< SystemMats_B.size(); i++){
    // uout -= learnparam_Re * SystemB * v
    MatMult(SystemMats_B[i], v, aux);
    VecAXPY(uout, -1.*paramsB[i], aux); 
    // vout += learnparam_Re * SystemB * u
    MatMult(SystemMats_B[i], u, aux);
    VecAXPY(vout, paramsB[i], aux);
  }
}


void HamiltonianModel::applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsH){

  /* Extract pointers to params that correspond to SystemMats_A vs _B */
  // Assume learnparams = [learnparams_re, learnparams_Im]
  double* paramsB = learnparamsH.data();                      // learnparams_re -> SystemMatsB
  double* paramsA = learnparamsH.data()+ SystemMats_B.size(); // learnparams_im -> SystemMatsA

  // Real parts of (-i * H)
  for (int i=0; i< SystemMats_A.size(); i++){
    // uout += learnparam_Im * SystemMat_A^T * u
    MatMultTranspose(SystemMats_A[i], u, aux);
    VecAXPY(uout, paramsA[i], aux); 
    // vout += learnparam_Im * SystemMat_A^T * v
    MatMultTranspose(SystemMats_A[i], v, aux);
    VecAXPY(vout, paramsA[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< SystemMats_B.size(); i++){
    // uout += learnparam_Re * SystemMat_B^T * v
    MatMultTranspose(SystemMats_B[i], v, aux);
    VecAXPY(uout, paramsB[i], aux); 
    // vout -= learnparam_Re * SystemMat_B^T * u
    MatMultTranspose(SystemMats_B[i], u, aux);
    VecAXPY(vout, -1.*paramsB[i], aux);
  }
}

void HamiltonianModel::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsH, int grad_skip){
  // gamma_bar_A += alpha * (  u^t sigma_A^t ubar + v^t sigma_A^t vbar )
  // gamma_bar_B += alpha * ( -v^t sigma_B^t ubar + u^t sigma_B^t vbar )

  double uAubar, vAvbar, vBubar, uBvbar;
  for (int i=0; i<SystemMats_B.size(); i++){
    MatMult(SystemMats_B[i], u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(SystemMats_B[i], v, aux); VecDot(aux, ubar, &vBubar);
    VecSetValue(grad, grad_skip + i, alpha*(-vBubar + uBvbar), ADD_VALUES);
  }  
  int skip = SystemMats_B.size();
  for (int i=0; i< SystemMats_A.size(); i++){
    MatMult(SystemMats_A[i], u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(SystemMats_A[i], v, aux); VecDot(aux, vbar, &vAvbar);
    VecSetValue(grad, grad_skip + i + skip, alpha*(uAubar + vAvbar), ADD_VALUES);
  }
}

void HamiltonianModel::printOperator(std::vector<double>& learnparamsH, std::string datadir){

  /* Create the Gellmann matrices*/
  std::vector<Mat> BasisMats_Re, BasisMats_Im;
  createGellmannMats(dim_rho, false, false, shifted_diag, false, BasisMats_Re, BasisMats_Im);

  /* Extract pointers to params that correspond to SystemMats_A vs _B */
  // Assume learnparams = [learnparams_re, learnparams_Im]
  double* learnparamsH_Re = learnparamsH.data();                      // learnparams_re -> SystemMatsB
  double* learnparamsH_Im = learnparamsH.data()+ SystemMats_B.size(); // learnparams_im -> SystemMatsA

  /* Assemble the Hamiltonian, MHz, H = \sum l_i*sigma_i */
  MatZeroEntries(Operator[0]);
  MatZeroEntries(Operator[1]);
  for (int i=0; i<BasisMats_Re.size(); i++) {
    MatAXPY(Operator[0], learnparamsH_Re[i] / (2.0*M_PI), BasisMats_Re[i], DIFFERENT_NONZERO_PATTERN);
  }
  for (int i=0; i<BasisMats_Im.size(); i++) {
    MatAXPY(Operator[1], learnparamsH_Im[i] / (2.0*M_PI), BasisMats_Im[i], DIFFERENT_NONZERO_PATTERN);
  }

  /* Clean up*/
  for (int i=0; i<BasisMats_Re.size(); i++) MatDestroy(&BasisMats_Re[i]);
  for (int i=0; i<BasisMats_Im.size(); i++) MatDestroy(&BasisMats_Im[i]);
  
  /* Print Hamiltonian to screen */
  printf("\nLearned Hamiltonian operator [MHz]: Re = \n");
  MatView(Operator[0], NULL);
  printf("Learned Hamiltonian operator [MHz]: Im = \n");
  MatView(Operator[1], NULL);

  /* Print Hamiltonian to files */
  char filename_re[254]; 
  char filename_im[254]; 
  snprintf(filename_re, 254, "%s/LearnedHamiltonian_Re.dat", datadir.c_str());
  snprintf(filename_im, 254, "%s/LearnedHamiltonian_Im.dat", datadir.c_str());
  PetscViewer viewer_re, viewer_im;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_re, &viewer_re);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_im, &viewer_im);
  MatView(Operator[0],viewer_re);
  MatView(Operator[1],viewer_im);
  PetscViewerDestroy(&viewer_re);
  PetscViewerDestroy(&viewer_im);
  printf("\nLearned Hamiltonian written to file %s, %s\n", filename_re, filename_im);

}
  
LindbladModel::LindbladModel(int dim_rho_, bool shifted_diag, bool upper_only, bool real_only) : UDEmodel(dim_rho_, LindbladType::BOTH) {
  dim_rho = dim_rho_;
  dim = dim_rho*dim_rho; 

  /* Assemble system Matrices */ 
  nbasis = createSystemMats(upper_only, real_only, shifted_diag);

  /* set the total number of learnable parameters */
  nparams = 0.5 * nbasis * (nbasis+1);

  /* Allocate storage for the summed-up lindblad system matrix */
  Operator.resize(1);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim, dim, NULL, &Operator[0]);
  MatSetUp(Operator[0]);
}

LindbladModel::~LindbladModel(){}


int LindbladModel::createSystemMats(bool upper_only, bool real_only, bool shifted_diag){
  /* Set up and store the Lindblad system matrices: 
  *   sigma_i.conj kron sigma_j - 1/2(I kron sigma_i^t sigma_j + (sigma_i^t sigma_j)^T kron I)
   * Here, all Basis mats are REAL (only using upper part of the real Gellmann mats), hence all go into A = Re(...)
   * Note that here we have: sigma.conj = sigma and (sigma^tsigma)^T
  */
  if (!real_only && dim_rho > 0) {
    printf("ERROR: Lindblad basis currently only implemented for real basis mats.");
    exit(1);
  }

  /* Create the Gellmann matrices*/
  std::vector<Mat> BasisMats_Re, BasisMats_Im;
  bool includeIdentity = true;
  createGellmannMats(dim_rho, upper_only, real_only, shifted_diag, includeIdentity, BasisMats_Re, BasisMats_Im);
  // createDecayBasis_2qubit(dim_rho, BasisMats_Re, includeIdentity);
  int nbasis = BasisMats_Re.size() - 1 + BasisMats_Im.size();

  // Note BasisMats[0] contains the identity. Grab it here:
  Mat Id = BasisMats_Re[0];

  for (int i=1; i<BasisMats_Re.size(); i++){ // loop starts at 1 to exclude Id
    for (int j=1; j<BasisMats_Re.size(); j++){

      Mat myMat, myMat1, myMat2, sigmasq;
      MatTransposeMatMult(BasisMats_Re[j], BasisMats_Re[i],MAT_INITIAL_MATRIX, PETSC_DEFAULT,&sigmasq);

      MatSeqAIJKron(BasisMats_Re[j], BasisMats_Re[i], MAT_INITIAL_MATRIX, &myMat);  // myMat = sigma_j \kron sigma_i
      MatSeqAIJKron(Id, sigmasq, MAT_INITIAL_MATRIX, &myMat1);                    // myMay1 = Id \kron sigma_i^tsigma_j
      MatAXPY(myMat, -0.5, myMat1, DIFFERENT_NONZERO_PATTERN);                    // myMat = sigma kron sigma - 0.5*(Id kron sigma^tsigma)
      MatTranspose(sigmasq, MAT_INPLACE_MATRIX, &sigmasq);
      MatSeqAIJKron(sigmasq, Id, MAT_INITIAL_MATRIX, &myMat2);  // myMat2 = (sigma_i^tsigma_j)^T \kron Id
      MatAXPY(myMat, -0.5, myMat2, DIFFERENT_NONZERO_PATTERN);  // myMat = sigma kron sigma - 0.5*( Id kron sigma^2 + sigma^2 kron Id)

      SystemMats_A.push_back(myMat);
      MatDestroy(&myMat1);
      MatDestroy(&myMat2);
      MatDestroy(&sigmasq);
    }
  }

  /* Clean up*/
  for (int i=0; i<BasisMats_Re.size(); i++) MatDestroy(&BasisMats_Re[i]);
  for (int i=0; i<BasisMats_Im.size(); i++) MatDestroy(&BasisMats_Im[i]);

  return nbasis;
}

void LindbladModel::applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL){

  // Real parts of lindblad terms
  for (int i=0; i<nbasis; i++){
    for (int j=0; j<nbasis; j++){
      int id_sys = i*nbasis + j;

      // Get aij coefficient
      double aij = 0.0;
      for (int l=0; l<nbasis; l++){
        double x_il = 0.0; 
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL[mapID(j,l)];
        aij += x_il * x_jl;
      }

      // uout += learnparam_Re * SystemA * u
      MatMult(SystemMats_A[id_sys], u, aux);
      VecAXPY(uout, aij, aux); 
      // vout += learnparam_Re * SystemA * v
      MatMult(SystemMats_A[id_sys], v, aux);
      VecAXPY(vout, aij, aux);

    }
  }
  // // Imaginary parts of lindblad terms
  // for (int i=0; i< SystemMats_B.size(); i++){
  //   printf("Should not be here.\n");
  //   exit(1);
  //   // uout -= learnparam_Im * SystemB * v
  //   MatMult(SystemMats_B[i], v, aux);
  //   VecAXPY(uout, -1.*learnparamsL_Im[i], aux); 
  //   // vout += learnparam_Im * SystemB * u
  //   MatMult(SystemMats_B[i], u, aux);
  //   VecAXPY(vout, learnparamsL_Im[i], aux);
  // }
}

void LindbladModel::applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL){
  // Real parts of lindblad terms
  for (int i=0; i<nbasis; i++){
    for (int j=0; j<nbasis; j++){
      int id_sys = i*nbasis + j;

      // Get aij coefficient
      double aij = 0.0;
      for (int l=0; l<nbasis; l++){
        double x_il = 0.0;
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL[mapID(j,l)];
        aij += x_il * x_jl;
      }

      // uout += learnparam_Re * SystemMat_A^T * u
      MatMultTranspose(SystemMats_A[id_sys], u, aux);
      VecAXPY(uout, aij, aux); 
      // vout += learnparam_Re * SystemMat_A^T * v
      MatMultTranspose(SystemMats_A[id_sys], v, aux);
      VecAXPY(vout, aij, aux);
    }
  }
  // // Imaginary parts of lindbladterms
  // for (int i=0; i< SystemMats_B.size(); i++){
  //   // uout += learnparam_Im * SystemMat_B^T * v
  //   MatMultTranspose(SystemMats_B[i], v, aux);
  //   VecAXPY(uout, learnparamsL_Im[i], aux); 
  //   // vout -= learnparam_Im * SystemMat_B^T * u
  //   MatMultTranspose(SystemMats_B[i], u, aux);
  //   VecAXPY(vout, -1.*learnparamsL_Im[i], aux);
  // }
}


void LindbladModel::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL, int grad_skip){
 double uAubar, vAvbar, vBubar, uBvbar;
  // Note: Storage in grad corresponds to x = [learn_H, learn_L], so need to skip to the second part of the gradient] 

  for (int i=0; i<nbasis; i++){
    for (int j=0; j<nbasis; j++){
      int id_sys = i*nbasis + j;

      MatMult(SystemMats_A[id_sys], u, aux); VecDot(aux, ubar, &uAubar);
      MatMult(SystemMats_A[id_sys], v, aux); VecDot(aux, vbar, &vAvbar);
      double aijbar = uAubar + vAvbar;

      for (int l=0; l<nbasis; l++){
        double x_il = 0.0;
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL[mapID(j,l)];
        double x_il_bar = x_jl * aijbar;
        double x_jl_bar = x_il * aijbar;
        if (i<=l) VecSetValue(grad, mapID(i,l)+grad_skip, alpha*x_il_bar, ADD_VALUES);
        if (j<=l) VecSetValue(grad, mapID(j,l)+grad_skip, alpha*x_jl_bar, ADD_VALUES);
      }
    }
  }
  // skip += SystemMats_A.size();
  // for (int i=0; i<SystemMats_B.size(); i++){
  //   MatMult(SystemMats_B[i], u, aux); VecDot(aux, vbar, &uBvbar);
  //   MatMult(SystemMats_B[i], v, aux); VecDot(aux, ubar, &vBubar);
  //   VecSetValue(grad, i+skip, alpha*(-vBubar + uBvbar), ADD_VALUES);
  // }
}

void LindbladModel::evalOperator(std::vector<double>& learnparamsL){

  /* Reset */
  MatZeroEntries(Operator[0]);

  /* Sum up the system operator */
  for (int i=0; i<nbasis; i++) {
    for (int j=0; j<nbasis; j++) {

      // Get aij coefficient
      double aij = 0.0;
      for (int l=0; l<nbasis; l++){
        double x_il = 0.0; 
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL[mapID(j,l)];
        aij += x_il * x_jl;
      }

      // Add to operator 
      int id_sys = i*nbasis + j;
      MatAXPY(Operator[0], aij, SystemMats_A[id_sys], DIFFERENT_NONZERO_PATTERN);
    }
  }
}

void LindbladModel::printOperator(std::vector<double>& learnparamsL, std::string datadir){

  if (dim_rho == 2) {
    // print coefficients to screen
    for (int i=0; i<nparams; i++){
      printf("Lindblad coeff %d: %1.8e\n", i, learnparamsL[i]);
    }
    printf(" -> maps to T_1 time %1.2f [us]\n", 1.0/learnparamsL[0]);
    printf(" -> maps to T_2 time %1.2f [us]\n", 1.0/(4.0*learnparamsL[1]));
  }

  /* assemble the system matrix */
  evalOperator(learnparamsL);

  // print to file
  char filename[254]; 
  snprintf(filename, 254, "%s/LearnedLindbladSystemMat.dat", datadir.c_str());
  PetscViewer viewer;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename, &viewer);
  MatView(Operator[0],viewer);
  PetscViewerDestroy(&viewer);
  printf("\nLearned Lindblad system matrix written to file %s\n", filename);
}