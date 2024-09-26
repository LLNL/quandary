#include "gellmannbasis.hpp"

GellmannBasis::GellmannBasis(int dim_rho_, bool upper_only_, bool shifted_diag_, LindbladType lindbladtype_){
  dim_rho = dim_rho_;
  lindbladtype = lindbladtype_;
  upper_only = upper_only_;
  shifted_diag = shifted_diag_;
  nparams = 0;

  // If Lindblad solver, dim = N^2, otherwise dim=N
  dim = dim_rho;
  if (lindbladtype != LindbladType::NONE){
    dim = dim_rho*dim_rho; 
  }

  /* First all offdiagonal matrices (re and im)*/
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      /* Real sigma_jk^re = |j><k| + |k><j| */ 
      Mat G_re;
      MatCreate(PETSC_COMM_WORLD, &G_re);
      MatSetType(G_re, MATSEQAIJ);
      MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
      MatSetUp(G_re);
      MatSetValue(G_re, j, k, 1.0, INSERT_VALUES);
      if (!upper_only) MatSetValue(G_re, k, j, 1.0, INSERT_VALUES);
      MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
      BasisMat_Re.push_back(G_re);

      /* Imaginary sigma_jk^im = -i|j><k| + i|k><j| */ 
      if (!upper_only) {
        Mat G_im;
        MatCreate(PETSC_COMM_WORLD, &G_im);
        MatSetType(G_im, MATSEQAIJ);
        MatSetSizes(G_im, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
        MatSetUp(G_im);
        MatSetValue(G_im, j, k, -1.0, INSERT_VALUES);
        MatSetValue(G_im, k, j, +1.0, INSERT_VALUES);
        /* Assemble and store */
        MatAssemblyBegin(G_im, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(G_im, MAT_FINAL_ASSEMBLY);
        BasisMat_Im.push_back(G_im);
      }
    }
  }

  /* Then all diagonal matrices (shifted, all real)  */
  for (int l=1; l<dim_rho; l++){
    Mat G_re;
    MatCreate(PETSC_COMM_WORLD, &G_re);
    MatSetType(G_re, MATSEQAIJ);
    MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(G_re);

    /* Diagonal mats 
     *  shifted:     sigma_l^Re = (2/(l(l+1))( sum_{j=l,...,N-1} -|j><j| - l|l><l|)
     *  not shifted: sigma_l^Re = (2/(l(l+1))( sum_{j=0,...,l-1}  |j><j| - l|l><l|) 
     */
    double factor = sqrt(2.0/(l*(l+1)));
    MatSetValue(G_re, l, l, -1.0*l*factor, ADD_VALUES);
    if (shifted_diag) {      
      for (int j=l; j<dim_rho; j++){
        MatSetValue(G_re, j, j, -1.0*factor, ADD_VALUES);
      }
    } else {  
      for (int j=0; j<l; j++){
        MatSetValue(G_re, j, j, factor, ADD_VALUES);
      }
    }
    MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
    BasisMat_Re.push_back(G_re);
  }

  /* Store the number of basis elements */
  nbasis = BasisMat_Re.size() + BasisMat_Im.size();

  /* set up the identity matrix */
  MatCreate(PETSC_COMM_WORLD, &Id);
  MatSetType(Id, MATSEQAIJ);
  MatSetSizes(Id, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
  for (int i=0; i<dim_rho; i++){
    MatSetValue(Id, i, i, 1.0, INSERT_VALUES);
  }
  MatAssemblyBegin(Id, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Id, MAT_FINAL_ASSEMBLY);

  /* Create an auxiliary vector for system matrix matmult */
  VecCreate(PETSC_COMM_WORLD, &aux);     // aux sized for Re(state) or Im(state) 
  VecSetSizes(aux , PETSC_DECIDE, dim);
  VecSetFromOptions(aux);
}


void GellmannBasis::showBasisMats(){

  printf("\n Here are all Gellmann Basis mats (shifted %d, upper_only %d):\n", shifted_diag, upper_only);
  printf("REAL Mats:\n");
  for (int i=0; i<BasisMat_Re.size(); i++){
    MatView(BasisMat_Re[i], NULL);
  }
  printf("REAL Mats:\n");
  for (int i=0; i<BasisMat_Im.size(); i++){
    MatView(BasisMat_Im[i], NULL);
  }
}

GellmannBasis::~GellmannBasis(){
  for (int i=0; i<BasisMat_Re.size(); i++){
    MatDestroy(&BasisMat_Re[i]);
  }
  for (int i=0; i<BasisMat_Im.size(); i++){
    MatDestroy(&BasisMat_Im[i]);
  }
  BasisMat_Re.clear();
  BasisMat_Im.clear();
  MatDestroy(&Id);
  VecDestroy(&aux);
  for (int i=0; i< SystemMats_A.size(); i++){
    MatDestroy(&SystemMats_A[i]);
  }
  for (int i=0; i<SystemMats_B.size(); i++){
    MatDestroy(&SystemMats_B[i]);
  }
  SystemMats_A.clear();
  SystemMats_B.clear();
}

StdGellmannBasis::StdGellmannBasis(int dim_rho_) : GellmannBasis(dim_rho_, false, false, LindbladType::BOTH) {}

StdGellmannBasis::~StdGellmannBasis(){}

HamiltonianBasis::HamiltonianBasis(int dim_rho_, bool shifted_diag_, LindbladType lindbladtype_) : GellmannBasis(dim_rho_, false, shifted_diag_, lindbladtype_) {

  /* Assemble system Matrices */
  assembleSystemMats();
  nparams = getNBasis();

  /* Allocate storage for returning learned Hamitonian operators */
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Re);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Im);
  MatSetUp(Operator_Re);
  MatSetUp(Operator_Im);
 }

HamiltonianBasis::~HamiltonianBasis(){

  MatDestroy(&Operator_Re);
  MatDestroy(&Operator_Im);
}

void HamiltonianBasis::assembleSystemMats(){

  /* Set up and store the Hamiltonian system matrices:
   *   (-i*sigma)   or vectorized   -i(I kron sigma - sigma^T kron I) 
   *  A = Re(-isigma)
   *  B = Im(-isigma)
   */

  // Set up -i*(Real_Gellmann), they go into Bd = Im(-iH)
  for (int i=0; i<BasisMat_Re.size(); i++){
    Mat myMat;
    if (lindbladtype == LindbladType::NONE){ // -sigma
      MatDuplicate(BasisMat_Re[i],  MAT_COPY_VALUES, &myMat);
      MatScale(myMat, -1.0);
    } else { // - I kron sigma + sigma^T kron I
      Mat myMat1, myMat2;
      MatTranspose(BasisMat_Re[i], MAT_INITIAL_MATRIX, &myMat1);      // myMat1 = sigma^T
      MatSeqAIJKron(myMat1, Id, MAT_INITIAL_MATRIX, &myMat);          // myMat = sigma^T kron I
      MatSeqAIJKron(Id, BasisMat_Re[i], MAT_INITIAL_MATRIX, &myMat2); // myMat2 = I kron sigma
      MatAXPY(myMat, -1.0, myMat2, DIFFERENT_NONZERO_PATTERN);        // myMat = sigma^T kron I - I kron sigma
      MatDestroy(&myMat1);
      MatDestroy(&myMat2);
    }
    SystemMats_B.push_back(myMat);
  }

  // Set up -i*(Imag_BasisMat), they go into Ad = Re(-iH) 
  Mat myMat;
  for (int i=0; i<BasisMat_Im.size(); i++){
    if (lindbladtype == LindbladType::NONE){ // sigma
      MatDuplicate(BasisMat_Im[i],  MAT_COPY_VALUES, &myMat);
    } else { // I kron sigma - sigma^T kron I
      Mat myMat1, myMat2;
      MatSeqAIJKron(Id, BasisMat_Im[i], MAT_INITIAL_MATRIX, &myMat); // myMat = I kron sigma
      MatTranspose(BasisMat_Im[i], MAT_INITIAL_MATRIX, &myMat1);    // myMat1 = sigma^T
      MatSeqAIJKron(myMat1, Id, MAT_INITIAL_MATRIX, &myMat2);       // myMat2 = sigma^T kron I
      MatAXPY(myMat, -1.0, myMat2, DIFFERENT_NONZERO_PATTERN);      // myMat = I kron sigma - sigma^T kron I
      MatDestroy(&myMat1);
      MatDestroy(&myMat2);
    }
    SystemMats_A.push_back(myMat);
  }
}


void HamiltonianBasis::applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsH_Re, std::vector<double>& learnparamsH_Im){
// Note: All purely real Basis Mats correspond to purely imaginary System matrices (stored in in SystemMatsB). Hence, the learnparams_Re (which go for BasisMats_Re), are applied to SystemMats_B, and learnparams_Im are applied to SystemMats_A!

  // Real parts of (-i * H)
  for (int i=0; i< SystemMats_A.size(); i++){
    // uout += learnparam_Im * SystemA * u
    MatMult(SystemMats_A[i], u, aux);
    VecAXPY(uout, learnparamsH_Im[i], aux); 
    // vout += learnparam_IM * SystemA * v
    MatMult(SystemMats_A[i], v, aux);
    VecAXPY(vout, learnparamsH_Im[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< SystemMats_B.size(); i++){
    // uout -= learnparam_Re * SystemB * v
    MatMult(SystemMats_B[i], v, aux);
    VecAXPY(uout, -1.*learnparamsH_Re[i], aux); 
    // vout += learnparam_Re * SystemB * u
    MatMult(SystemMats_B[i], u, aux);
    VecAXPY(vout, learnparamsH_Re[i], aux);
  }
}


void HamiltonianBasis::applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsH_Re, std::vector<double>& learnparamsH_Im){
// Note: All purely real Basis Mats correspond to purely imaginary System matrices (stored in in SystemMatsB). Hence, the learnparams_Re (which go for BasisMats_Re), are applied to SystemMats_B, and learnparams_Im are applied to SystemMats_A!

  // Real parts of (-i * H)
  for (int i=0; i< SystemMats_A.size(); i++){
    // uout += learnparam_Im * SystemMat_A^T * u
    MatMultTranspose(SystemMats_A[i], u, aux);
    VecAXPY(uout, learnparamsH_Im[i], aux); 
    // vout += learnparam_Im * SystemMat_A^T * v
    MatMultTranspose(SystemMats_A[i], v, aux);
    VecAXPY(vout, learnparamsH_Im[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< SystemMats_B.size(); i++){
    // uout += learnparam_Re * SystemMat_B^T * v
    MatMultTranspose(SystemMats_B[i], v, aux);
    VecAXPY(uout, learnparamsH_Re[i], aux); 
    // vout -= learnparam_Re * SystemMat_B^T * u
    MatMultTranspose(SystemMats_B[i], u, aux);
    VecAXPY(vout, -1.*learnparamsH_Re[i], aux);
  }
}

void HamiltonianBasis::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL_Re, int skipID){
// Note: learnparam storage is [learn_Re, learn_Im]. For the Hamiltonian, learn_Re is applied to System_A and learn_Im is applied to System_B. Hence, invert order here: First part of the gradient goes with System_B (learn_Re), and second goes with System_A (learn_Im).

  // gamma_bar_A += alpha * (  u^t sigma_A^t ubar + v^t sigma_A^t vbar )
  // gamma_bar_B += alpha * ( -v^t sigma_B^t ubar + u^t sigma_B^t vbar )

  double uAubar, vAvbar, vBubar, uBvbar;
  for (int i=0; i<SystemMats_B.size(); i++){
    MatMult(SystemMats_B[i], u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(SystemMats_B[i], v, aux); VecDot(aux, ubar, &vBubar);
    VecSetValue(grad, i, alpha*(-vBubar + uBvbar), ADD_VALUES);
  }  
  int skip = SystemMats_B.size();
  for (int i=0; i< SystemMats_A.size(); i++){
    MatMult(SystemMats_A[i], u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(SystemMats_A[i], v, aux); VecDot(aux, vbar, &vAvbar);
    VecSetValue(grad, i+skip, alpha*(uAubar + vAvbar), ADD_VALUES);
  }
}

void HamiltonianBasis::printOperator(std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im, std::string datadir){

  /* Assemble the Hamiltonian, MHz, H = \sum l_i*sigma_i */
  MatZeroEntries(Operator_Re);
  MatZeroEntries(Operator_Im);
  for (int i=0; i<getNBasis_Re(); i++) {
    MatAXPY(Operator_Re, learnparams_Re[i] / (2.0*M_PI), getBasisMat_Re(i), DIFFERENT_NONZERO_PATTERN);
  }
  for (int i=0; i<getNBasis_Im(); i++) {
    MatAXPY(Operator_Im, learnparams_Im[i] / (2.0*M_PI), getBasisMat_Im(i), DIFFERENT_NONZERO_PATTERN);
  }
  // // If diagonally shifted: H -= H_00*Id */
  // if (shift_diag) {
  //   double h00=0.0;
  //   MatGetValue(Operator_Re, 0, 0, &h00);
  //   MatShift(Operator_Re, -h00);
  // }

  /* Print to screen */
  printf("\nLearned Hamiltonian operator [MHz]: Re = \n");
  MatView(Operator_Re, NULL);
  printf("Learned Hamiltonian operator [MHz]: Im = \n");
  MatView(Operator_Im, NULL);

  /* print to file */
  char filename_re[254]; 
  char filename_im[254]; 
  snprintf(filename_re, 254, "%s/LearnedHamiltonian_Re.dat", datadir.c_str());
  snprintf(filename_im, 254, "%s/LearnedHamiltonian_Im.dat", datadir.c_str());
  PetscViewer viewer_re, viewer_im;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_re, &viewer_re);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename_im, &viewer_im);
  MatView(Operator_Re,viewer_re);
  MatView(Operator_Im,viewer_im);
  PetscViewerDestroy(&viewer_re);
  PetscViewerDestroy(&viewer_im);
  printf("\nLearned Hamiltonian written to file %s, %s\n", filename_re, filename_im);
}
  
LindbladBasis::LindbladBasis(int dim_rho_, bool shifted_diag_) : GellmannBasis(dim_rho_, true, shifted_diag_, LindbladType::BOTH) {

  /* Assemble system Matrices */
  assembleSystemMats();
  nparams = 0.5 * getNBasis() * (getNBasis()+1);

  /* Allocate storage for the summed-up lindblad system matrix */
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim, dim, NULL, &Operator);
  MatSetUp(Operator);
  // MatCreate(PETSC_COMM_WORLD, &Operator);
  // MatSetType(Operator, MATSEQAIJ);
  // MatSetSizes(Operator, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  // MatSetUp(Operator);
  // MatAssemblyBegin(Operator, MAT_FINAL_ASSEMBLY);
  // MatAssemblyEnd(Operator, MAT_FINAL_ASSEMBLY);
}

LindbladBasis::~LindbladBasis(){}


void LindbladBasis::assembleSystemMats(){
  /* Set up and store the Lindblad system matrices: 
  *   sigma_i.conj kron sigma_j - 1/2(I kron sigma_i^t sigma_j + (sigma_i^t sigma_j)^T kron I)
   * Here, all Basis mats are REAL (only using upper part of the real Gellmann mats), hence all go into A = Re(...)
   * Note that here we have: sigma.conj = sigma and (sigma^tsigma)^T
  */

  for (int i=0; i<BasisMat_Re.size(); i++){
    for (int j=0; j<BasisMat_Re.size(); j++){

      Mat myMat, myMat1, myMat2, sigmasq;
      MatTransposeMatMult(BasisMat_Re[i], BasisMat_Re[j],MAT_INITIAL_MATRIX, PETSC_DEFAULT,&sigmasq);

      MatSeqAIJKron(BasisMat_Re[i], BasisMat_Re[j], MAT_INITIAL_MATRIX, &myMat);  // myMat = sigma_i \kron sigma_j
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
}

void LindbladBasis::applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL_Re, std::vector<double>& learnparamsL_Im){

  // evalOperator(learnparamsL_Re);
  // // uout += learnparam_Re * SystemA * u
  // MatMult(Operator, u, aux);
  // VecAXPY(uout, 1.0, aux); 
  // // vout += learnparam_Re * SystemA * v
  // MatMult(Operator, v, aux);
  // VecAXPY(vout, 1.0, aux);
 
  // Real parts of lindblad terms
  for (int i=0; i< BasisMat_Re.size(); i++){
    for (int j=0; j< BasisMat_Re.size(); j++){
      int id_sys = i*BasisMat_Re.size() + j;

      // Get aij coefficient
      double aij = 0.0;
      for (int l=0; l<BasisMat_Re.size(); l++){
        double x_il = 0.0; 
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL_Re[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL_Re[mapID(j,l)];
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

void LindbladBasis::applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL_Re, std::vector<double>& learnparamsL_Im){
  // Real parts of lindblad terms
  for (int i=0; i< BasisMat_Re.size(); i++){
    for (int j=0; j< BasisMat_Re.size(); j++){
      int id_sys = i*BasisMat_Re.size() + j;

      // Get aij coefficient
      double aij = 0.0;
      for (int l=0; l<BasisMat_Re.size(); l++){
        double x_il = 0.0;
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL_Re[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL_Re[mapID(j,l)];
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


void LindbladBasis::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL_Re, int skip){
 double uAubar, vAvbar, vBubar, uBvbar;
  // Note: Storage in grad corresponds to x = [learn_H, learn_L], so need to skip to the second part of the gradient] 

  for (int i=0; i< BasisMat_Re.size(); i++){
    for (int j=0; j< BasisMat_Re.size(); j++){
      int id_sys = i*BasisMat_Re.size() + j;

      MatMult(SystemMats_A[id_sys], u, aux); VecDot(aux, ubar, &uAubar);
      MatMult(SystemMats_A[id_sys], v, aux); VecDot(aux, vbar, &vAvbar);
      double aijbar = uAubar + vAvbar;

      for (int l=0; l<BasisMat_Re.size(); l++){
        double x_il = 0.0;
        double x_jl = 0.0;
        if (i<=l) x_il = learnparamsL_Re[mapID(i,l)];
        if (j<=l) x_jl = learnparamsL_Re[mapID(j,l)];
        double x_il_bar = x_jl * aijbar;
        double x_jl_bar = x_il * aijbar;
        if (i<=l) VecSetValue(grad, mapID(i,l)+skip, alpha*x_il_bar, ADD_VALUES);
        if (j<=l) VecSetValue(grad, mapID(j,l)+skip, alpha*x_jl_bar, ADD_VALUES);
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

void LindbladBasis::evalOperator(std::vector<double>& learnparams_Re){

  /* Reset */
  MatZeroEntries(Operator);

  /* Sum up the system operator */
  for (int i=0; i<getNBasis_Re(); i++) {
    for (int j=0; j<getNBasis_Re(); j++) {

      // Get aij coefficient
      double aij = 0.0;
      for (int l=0; l<BasisMat_Re.size(); l++){
        double x_il = 0.0; 
        double x_jl = 0.0;
        if (i<=l) x_il = learnparams_Re[mapID(i,l)];
        if (j<=l) x_jl = learnparams_Re[mapID(j,l)];
        aij += x_il * x_jl;
      }

      // Add to operator 
      int id_sys = i*getNBasis_Re() + j;
      MatAXPY(Operator, aij, SystemMats_A[id_sys], DIFFERENT_NONZERO_PATTERN);
    }
  }
}

void LindbladBasis::printOperator(std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im, std::string datadir){
  assert(getSystemMats_B().size() == 0);

  if (dim_rho == 2) {
    // print coefficients to screen
    for (int i=0; i<getNParams(); i++){
      printf("Lindblad coeff %d: %1.8e\n", i, learnparams_Re[i]);
    }
    printf(" -> maps to T_1 time %1.2f [us]\n", 1.0/learnparams_Re[0]);
    printf(" -> maps to T_2 time %1.2f [us]\n", 1.0/(4.0*learnparams_Re[1]));
  }

  /* assemble the system matrix */
  evalOperator(learnparams_Re);

  // print to file
  char filename[254]; 
  snprintf(filename, 254, "%s/LearnedLindbladSystemMat.dat", datadir.c_str());
  PetscViewer viewer;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename, &viewer);
  MatView(Operator,viewer);
  printf("\nLearned Lindblad system matrix written to file %s\n", filename);

  PetscViewerDestroy(&viewer);
  MatDestroy(&Operator);
}