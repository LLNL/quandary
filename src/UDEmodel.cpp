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
  for (int i=0; i<Operator_Re.size(); i++) MatDestroy(&Operator_Re[i]);
  for (int i=0; i<Operator_Im.size(); i++) MatDestroy(&Operator_Im[i]);
}

HamiltonianModel::HamiltonianModel(int dim_rho_, bool shifted_diag_, LindbladType lindbladtype) : UDEmodel(dim_rho_, lindbladtype) {
  shifted_diag = shifted_diag_;

  /* Assemble system Matrices */
  createSystemMats(lindbladtype);

  /* Set the number of learnable parameters */
  nparams = SystemMats_A.size() + SystemMats_B.size();

  if (dim_rho_ <= 0) return;

  /* Allocate storage for returning learned Hamitonian operators (Real and imaginary parts) */
  Operator_Re.resize(1);
  Operator_Im.resize(1);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Re[0]);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Im[0]);
  MatSetUp(Operator_Re[0]);
  MatSetUp(Operator_Im[0]);
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

void HamiltonianModel::writeOperator(std::vector<double>& learnparamsH, std::string datadir){

  /* Create the Gellmann matrices*/
  std::vector<Mat> BasisMats_Re, BasisMats_Im;
  createGellmannMats(dim_rho, false, false, shifted_diag, false, BasisMats_Re, BasisMats_Im);

  /* Extract pointers to params that correspond to SystemMats_A vs _B */
  // Assume learnparams = [learnparams_re, learnparams_Im]
  double* learnparamsH_Re = learnparamsH.data();                      // learnparams_re -> SystemMatsB
  double* learnparamsH_Im = learnparamsH.data()+ SystemMats_B.size(); // learnparams_im -> SystemMatsA

  /* Assemble the Hamiltonian, MHz, H = \sum l_i*sigma_i */
  MatZeroEntries(Operator_Re[0]);
  MatZeroEntries(Operator_Im[0]);
  for (int i=0; i<BasisMats_Re.size(); i++) {
    MatAXPY(Operator_Re[0], learnparamsH_Re[i] / (2.0*M_PI), BasisMats_Re[i], DIFFERENT_NONZERO_PATTERN);
  }
  for (int i=0; i<BasisMats_Im.size(); i++) {
    MatAXPY(Operator_Im[0], learnparamsH_Im[i] / (2.0*M_PI), BasisMats_Im[i], DIFFERENT_NONZERO_PATTERN);
  }

  /* Clean up*/
  for (int i=0; i<BasisMats_Re.size(); i++) MatDestroy(&BasisMats_Re[i]);
  for (int i=0; i<BasisMats_Im.size(); i++) MatDestroy(&BasisMats_Im[i]);
  
  // /* Print Hamiltonian to screen */
  // printf("\nLearned Hamiltonian operator [MHz]: Re = \n");
  // MatView(Operator_Re[0], NULL);
  // printf("Learned Hamiltonian operator [MHz]: Im = \n");
  // MatView(Operator_Im[0], NULL);

  /* Print Hamiltonian to files */
  char filename_re[254]; 
  char filename_im[254]; 
  snprintf(filename_re, 254, "%s/LearnedHamiltonian_Re.dat", datadir.c_str());
  snprintf(filename_im, 254, "%s/LearnedHamiltonian_Im.dat", datadir.c_str());
  PetscViewer viewer_re, viewer_im;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_re, &viewer_re);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_im, &viewer_im);
  MatView(Operator_Re[0],viewer_re);
  MatView(Operator_Im[0],viewer_im);
  PetscViewerDestroy(&viewer_re);
  PetscViewerDestroy(&viewer_im);
  printf("\nLearned Hamiltonian written to file %s, %s\n", filename_re, filename_im);

}
  
LindbladModel::LindbladModel(int dim_rho_, bool shifted_diag, bool upper_only, bool real_only_) : UDEmodel(dim_rho_, LindbladType::BOTH) {
  real_only = real_only_;
  dim_rho = dim_rho_;
  dim = dim_rho*dim_rho; 

  /* Assemble system Matrices */ 
  nbasis = createSystemMats(upper_only, real_only, shifted_diag);

  /* set the total number of learnable parameters */
#if DOUBLESUM
    int nparams_re = 0.5 * nbasis* (nbasis+1);
    nparams = nparams_re;
    if (!real_only){
      nparams = 2*nparams_re;   // storing first all real parts, then all imaginary parts
    } 
#else
    nparams = nbasis;
#endif 

  if (dim_rho <= 0) return;

  /* Allocate storage for the summed-up lindblad operators (L_alpha) */
  // Number of operators = N^2-1
  Operator_Re.resize(dim-1);
  for (int i=0; i<Operator_Re.size(); i++){
    MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Re[i]);
    MatSetUp(Operator_Re[i]);
  }
  if (!real_only) {
    Operator_Im.resize(dim-1);
    for (int i=0; i<Operator_Im.size(); i++){
      MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Im[i]);
      MatSetUp(Operator_Im[i]);
    }
  }
}

LindbladModel::~LindbladModel(){}

void LindbladModel::getCoeffIJ(int i, int j, std::vector<double>& learnparamsL, double* aij_re_out, double* aij_im_out){

  /* Skip to imaginary parts */
  int idIm = learnparamsL.size()/2;

  /* Sum up the Aij coefficient (complex!) */
  double aij_re = 0.0;
  double aij_im = 0.0;
  for (int l=0; l<nbasis; l++){
    double x_re_il = 0.0; 
    double x_re_jl = 0.0;
    double x_im_il = 0.0; 
    double x_im_jl = 0.0;
    if (l<=i) {
      x_re_il = learnparamsL[mapID(l,i)];
      if (!real_only) x_im_il = learnparamsL[idIm + mapID(l,i)];
    }
    if (l<=j) {
      x_re_jl = learnparamsL[mapID(l,j)];
      if (!real_only) x_im_jl = learnparamsL[idIm + mapID(l,j)];
    }
    aij_re += x_re_il * x_re_jl + x_im_il * x_im_jl;
    aij_im += x_im_il * x_re_jl - x_re_il * x_im_jl;
  }
 
  /* Return */
  *aij_re_out = aij_re;
  *aij_im_out = aij_im;
}

int LindbladModel::createSystemMats(bool upper_only, bool real_only, bool shifted_diag){
  /* Set up and store the Lindblad system matrices: 
  *   E_i.conj kron E_j - 1/2(I kron E_i^t E_j+ (E_i^t E_j)^T kron I)
   * Real Basis mats A = Re(...), Imaginary Basis mats go into Im(...)
  */

  /* Create the Basis matrices*/
  std::vector<Mat> BasisMats_Re, BasisMats_Im;
  bool includeIdentity = true;
  // createGellmannMats(dim_rho, upper_only, real_only, shifted_diag, includeIdentity, BasisMats_Re, BasisMats_Im);
  createEijBasisMats(dim_rho, includeIdentity, BasisMats_Re, BasisMats_Im);
  // createDecayBasis_2qubit(dim_rho, BasisMats_Re, includeIdentity);
  int nbasis = BasisMats_Re.size() - 1 + BasisMats_Im.size();
  if (nbasis < 0) nbasis = 0;

  // Note BasisMats[0] contains the identity. Grab it here:
  Mat Id = BasisMats_Re[0];

  for (int i=1; i<BasisMats_Re.size(); i++){ // loop starts at 1 to exclude Id
#if DOUBLESUM
    for (int j=1; j<BasisMats_Re.size(); j++){
#else 
    int j=i;
#endif
      Mat myMat, myMat_Im, myMat1, myMat2, sigmasq;
      MatTransposeMatMult(BasisMats_Re[j], BasisMats_Re[i],MAT_INITIAL_MATRIX, PETSC_DEFAULT,&sigmasq);

      MatSeqAIJKron(BasisMats_Re[j], BasisMats_Re[i], MAT_INITIAL_MATRIX, &myMat);  // myMat = sigma_j \kron sigma_i
      MatSeqAIJKron(Id, sigmasq, MAT_INITIAL_MATRIX, &myMat1);                    // myMay1 = Id \kron sigma_i^tsigma_j
      MatAXPY(myMat, -0.5, myMat1, DIFFERENT_NONZERO_PATTERN);                    // myMat = sigma kron sigma - 0.5*(Id kron sigma^tsigma)
      MatTranspose(sigmasq, MAT_INPLACE_MATRIX, &sigmasq);
      MatSeqAIJKron(sigmasq, Id, MAT_INITIAL_MATRIX, &myMat2);  // myMat2 = (sigma_i^tsigma_j)^T \kron Id
      MatAXPY(myMat, -0.5, myMat2, DIFFERENT_NONZERO_PATTERN);  // myMat = sigma kron sigma - 0.5*( Id kron sigma^2 + sigma^2 kron Id)

      SystemMats_A.push_back(myMat);
      MatDuplicate(myMat, MAT_COPY_VALUES, &myMat_Im);
      SystemMats_B.push_back(myMat_Im);
      MatDestroy(&myMat1);
      MatDestroy(&myMat2);
      MatDestroy(&sigmasq);
#if DOUBLESUM
    }
#endif
  }

  // printf("Real\n");
  // for (int i=0; i<SystemMats_A.size(); i++){
  //   MatView(SystemMats_A[i], NULL);
  // }
  // printf("Imag\n");
  // for (int i=0; i<SystemMats_B.size(); i++){
  //   MatView(SystemMats_B[i], NULL);
  // }
  // exit(1);

  /* Clean up*/
  for (int i=0; i<BasisMats_Re.size(); i++) MatDestroy(&BasisMats_Re[i]);
  for (int i=0; i<BasisMats_Im.size(); i++) MatDestroy(&BasisMats_Im[i]);

  return nbasis;
}

void LindbladModel::applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL){
  // learnparamsL = [params_REAL, params_IMAL]

  // Mat Tr, Ti;
  // MatCreate(PETSC_COMM_WORLD, &Tr);
  // MatCreate(PETSC_COMM_WORLD, &Ti);
  // MatSetType(Tr, MATSEQAIJ);
  // MatSetType(Ti, MATSEQAIJ);
  // int nT = dim_rho*dim_rho - 1;
  // MatSetSizes(Tr, PETSC_DECIDE, PETSC_DECIDE, nT, nT);
  // MatSetSizes(Ti, PETSC_DECIDE, PETSC_DECIDE, nT, nT);
  // MatSetUp(Tr);
  // MatSetUp(Ti);
 

#if DOUBLESUM
  for (int i=0; i<nbasis; i++){
    for (int j=0; j<nbasis; j++){
      int id_sys = i*nbasis + j;
      
      // Get Aij coefficients
      double aij_re = 0.0;
      double aij_im = 0.0;
      getCoeffIJ(i,j, learnparamsL, &aij_re, &aij_im);
      // MatSetValue(Tr, i,j, aij_re, INSERT_VALUES);
      // MatSetValue(Ti, i,j, aij_im, INSERT_VALUES);

#else
  for (int i=0; i<SystemMats_A.size(); i++){
      int id_sys = i;
      // Get aij coefficient
      int idIm = learnparamsL.size()/2;
      double aij_re = learnparamsL[i];
      double aij_im = learnparamsL[idIm + i];
#endif 
      /* Apply real system mat */
      if (fabs(aij_re) > 1e-14) {
        // uout += learnparam_Re * SystemA * u
        MatMult(SystemMats_A[id_sys], u, aux);
        VecAXPY(uout, aij_re, aux); 
        // vout += learnparam_Re * SystemA * v
        MatMult(SystemMats_A[id_sys], v, aux);
        VecAXPY(vout, aij_re, aux);
      }

      /* Apply imag system mat */
      if (fabs(aij_im) > 1e-14) {
        // uout -= learnparam_Im * SystemB * v
        MatMult(SystemMats_B[id_sys], v, aux);
        VecAXPY(uout, -1.0*aij_im, aux); 
        // vout += learnparam_im * systemb * u
        MatMult(SystemMats_B[id_sys], u, aux);
        VecAXPY(vout, aij_im, aux);
      }
#if DOUBLESUM
    }
#endif
  }

  // MatAssemblyBegin(Tr, MAT_FINAL_ASSEMBLY);
  // MatAssemblyBegin(Ti, MAT_FINAL_ASSEMBLY);
  // MatAssemblyEnd(Tr, MAT_FINAL_ASSEMBLY);
  // MatAssemblyEnd(Ti, MAT_FINAL_ASSEMBLY);

  // // Test hermitianity Tr = Tr^T, and Ti = -Ti^T:
  // PetscBool issymm;
  // MatIsSymmetric(Tr, 1e-16, &issymm);
  // Mat Tid;
  // MatTranspose(Ti, MAT_INITIAL_MATRIX, &Tid);
  // MatAXPY(Tid, 1.0, Ti, DIFFERENT_NONZERO_PATTERN);
  // double norm = 0.0;
  // MatNorm(Tid, NORM_FROBENIUS, &norm);
  // printf("Xre is symm? %d\n", issymm);
  // printf("Xim is antisymm? 0==%1.4e\n", norm);
  // MatView(Tr, NULL);
  // MatView(Ti, NULL);
  // exit(1);

}

void LindbladModel::applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL){
#if DOUBLESUM
  for (int i=0; i<nbasis; i++){
    for (int j=0; j<nbasis; j++){
      int id_sys = i*nbasis + j;
      // Get aij coefficient
      double aij_re = 0.0;
      double aij_im = 0.0;
      getCoeffIJ(i,j, learnparamsL, &aij_re, &aij_im);
#else
  for (int i=0; i<SystemMats_A.size();i++){
    int j=i;
    int id_sys = i;

    // Get aij coefficient
    int idIm = learnparamsL.size()/2;
    double aij_re = learnparamsL[i];
    double aij_im = learnparamsL[idIm + i];
#endif
      /* Real part */
      if (fabs(aij_re) > 1e-14) {
        // uout += learnparam_Re * SystemMat_A^T * u
        MatMultTranspose(SystemMats_A[id_sys], u, aux);
        VecAXPY(uout, aij_re, aux); 
        // vout += learnparam_Re * SystemMat_A^T * v
        MatMultTranspose(SystemMats_A[id_sys], v, aux);
        VecAXPY(vout, aij_re, aux);
      }
      /* Imag part */
      if (fabs(aij_im) > 1e-14) {
        // uout += learnparam_Re * SystemMat_B^T * v
        MatMultTranspose(SystemMats_B[id_sys], v, aux);
        VecAXPY(uout, aij_im, aux); 
        // vout -= learnparam_Re * SystemMat_B^T * u
        MatMultTranspose(SystemMats_B[id_sys], u, aux);
        VecAXPY(vout, -1.*aij_im, aux);
      }
#if DOUBLESUM
    }
#endif
  }
}


void LindbladModel::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL, int grad_skip){
 double uAubar, vAvbar, vBubar, uBvbar;
  // Note: Storage in grad corresponds to x = [learn_H, learn_L], the input 'grad_skip' skips to this second part of the gradient corresponding to learnparamsL] 
    
  int idIm = learnparamsL.size()/2;
#if DOUBLESUM
  for (int i=0; i<nbasis; i++){
    for (int j=0; j<nbasis; j++){
      int id_sys = i*nbasis + j;

      // Reals 
      MatMult(SystemMats_A[id_sys], u, aux); VecDot(aux, ubar, &uAubar);
      MatMult(SystemMats_A[id_sys], v, aux); VecDot(aux, vbar, &vAvbar);
      double aijbar_re = uAubar + vAvbar;
      // Imags
      MatMult(SystemMats_B[id_sys], u, aux); VecDot(aux, vbar, &uBvbar);
      MatMult(SystemMats_B[id_sys], v, aux); VecDot(aux, ubar, &vBubar);
      double aijbar_im = - vBubar + uBvbar;

      for (int l=0; l<nbasis; l++){
        double x_re_il = 0.0;
        double x_re_jl = 0.0;
        double x_im_il = 0.0;
        double x_im_jl = 0.0;
        if (l<=i) {
          x_re_il = learnparamsL[mapID(l,i)];
          if (!real_only) x_im_il = learnparamsL[idIm + mapID(l,i)];
        }
        if (l<=j) {
          x_re_jl = learnparamsL[mapID(l,j)];
          if (!real_only) x_im_jl = learnparamsL[idIm + mapID(l,j)];
        }
        // FWD:
        // aij_re += x_re_il * x_re_jl + x_im_il * x_im_jl;
        // aij_im += x_im_il * x_re_jl - x_re_il * x_im_jl;
        double x_re_il_bar = x_re_jl * aijbar_re - x_im_jl * aijbar_im;
        double x_re_jl_bar = x_re_il * aijbar_re + x_im_il * aijbar_im;
        double x_im_il_bar = x_im_jl * aijbar_re + x_re_jl * aijbar_im;
        double x_im_jl_bar = x_im_il * aijbar_re - x_re_il * aijbar_im;
        if (l<=i) {
          VecSetValue(grad, mapID(l,i)+grad_skip, alpha*x_re_il_bar, ADD_VALUES);
          if (!real_only) VecSetValue(grad, idIm + mapID(l,i)+grad_skip, alpha*x_im_il_bar, ADD_VALUES);
        }
        if (l<=j) {
          VecSetValue(grad, mapID(l,j)+grad_skip, alpha*x_re_jl_bar, ADD_VALUES);
          if (!real_only) VecSetValue(grad, idIm + mapID(l,j)+grad_skip, alpha*x_im_jl_bar, ADD_VALUES);
        }
      }
    }
  }
#else
  for (int i=0; i<SystemMats_A.size(); i++){
    int j=i;
    int id_sys = i;

    MatMult(SystemMats_A[id_sys], u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(SystemMats_A[id_sys], v, aux); VecDot(aux, vbar, &vAvbar);
    double aijbar_re = uAubar + vAvbar;
    MatMult(SystemMats_B[id_sys], u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(SystemMats_B[id_sys], v, aux); VecDot(aux, ubar, &vBubar);
    double aijbar_im = - vBubar + uBvbar;

    VecSetValue(grad, i+grad_skip, alpha*aijbar_re, ADD_VALUES);
    VecSetValue(grad, idIm + i+grad_skip, alpha*aijbar_im, ADD_VALUES);
  }
#endif

}

void LindbladModel::evalOperator(std::vector<double>& learnparamsL){

  /* Reset */
  for (int i=0; i<Operator_Re.size(); i++) {
    MatZeroEntries(Operator_Re[i]);
  }
  for (int i=0; i<Operator_Im.size(); i++) {
    MatZeroEntries(Operator_Im[i]);
  }

  /* Sum up the Lindblad operator */
  std::vector<Mat> BasisMats_Re, BasisMats_Im;
  createEijBasisMats(dim_rho, false, BasisMats_Re, BasisMats_Im);

#if DOUBLESUM
  int idIm = learnparamsL.size()/2;
  for (int l=0; l<nbasis; l++) {// Iterats over operators L_l
    for (int k=0; k<nbasis; k++){ // Iterates over Basis expansion E_k, coeff x_k^l

      // Get coefficient x_k^l from column l of lower-triangular X matrix
      double xk_re = 0.0; 
      double xk_im = 0.0; 
      if (l<=k) {
        xk_re = learnparamsL[mapID(l,k)];
        if (!real_only) xk_im = learnparamsL[idIm + mapID(l,k)];
      }

      // Add to operator 
      MatAXPY(Operator_Re[l], xk_re, BasisMats_Re[k], DIFFERENT_NONZERO_PATTERN);
      if (!real_only) MatAXPY(Operator_Im[l], xk_im, BasisMats_Re[k], DIFFERENT_NONZERO_PATTERN);
    }
  }
#else 
  int idIm = learnparamsL.size()/2;
  for (int i=0; i<SystemMats_A.size(); i++){
    // Get aij coefficient and add to operator
    double aij_re = learnparamsL[i];
    double aij_im = learnparamsL[idIm + i];
      MatAXPY(Operator[0], aij_re, SystemMats_A[i], DIFFERENT_NONZERO_PATTERN);
      MatAXPY(Operator[1], aij_im, SystemMats_B[i], DIFFERENT_NONZERO_PATTERN);
  }
#endif
  /* Destroy the basis matrices */
  for (int i=0; i<BasisMats_Re.size(); i++) MatDestroy(&BasisMats_Re[i]);
  for (int i=0; i<BasisMats_Im.size(); i++) MatDestroy(&BasisMats_Im[i]);
  BasisMats_Re.clear();
  BasisMats_Im.clear();

}

void LindbladModel::writeOperator(std::vector<double>& learnparamsL, std::string datadir){

  // if (dim_rho == 2) {
  //   // print coefficients to screen
  //   for (int i=0; i<nparams; i++){
  //     printf("Lindblad coeff %d: %1.8e\n", i, learnparamsL[i]);
  //   }
  //   printf(" -> maps to T_1 time %1.2f [us]\n", 1.0/learnparamsL[0]);
  //   printf(" -> maps to T_2 time %1.2f [us]\n", 1.0/(4.0*learnparamsL[1]));
  // }

  /* assemble the system matrix */
  evalOperator(learnparamsL);

  // print to file
  char filename[254]; 
  char filename_im[254]; 
  snprintf(filename, 254, "%s/LearnedLindbladOperators_Re.dat", datadir.c_str());
  snprintf(filename_im, 254, "%s/LearnedLindbladOperators_Im.dat", datadir.c_str());
  PetscViewer viewer, viewer_im;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename, &viewer);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_im, &viewer_im);
  for (int l=0; l<Operator_Re.size(); l++) {
    MatView(Operator_Re[l],viewer);
  }
  for (int l=0; l<Operator_Im.size(); l++) {
    MatView(Operator_Im[l],viewer_im);
  }
  PetscViewerDestroy(&viewer);
  PetscViewerDestroy(&viewer_im);
  printf("\nLearned Lindblad system matrix written to file %s and %s\n", filename, filename_im);
}