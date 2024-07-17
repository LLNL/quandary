#include "learning.hpp"

GellmannBasis::GellmannBasis(int dim_rho_, bool upper_only_, LindbladType lindbladtype_){
  dim_rho = dim_rho_;
  lindbladtype = lindbladtype_;
  upper_only = upper_only_;

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

    /* shifted diagonal mats: sigma_l^Re = (2/(l(l+1))( sum_j -|j><j| - l|l><l|) */
    double factor = sqrt(2.0/(l*(l+1)));
    MatSetValue(G_re, l, l, -1.0*l*factor, ADD_VALUES);
    for (int j=l; j<dim_rho; j++){
      MatSetValue(G_re, j, j, -1.0*factor, ADD_VALUES);
    }
    MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
    BasisMat_Re.push_back(G_re);
  }

  /* Store the number of basis elements */
  nbasis = BasisMat_Re.size() + BasisMat_Im.size();

  /* Create an auxiliary vector for system matrix matmult */
  VecCreate(PETSC_COMM_WORLD, &aux);     // aux sized for Re(state) or Im(state) 
  VecSetSizes(aux , PETSC_DECIDE, dim);
  VecSetFromOptions(aux);
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

HamiltonianBasis::HamiltonianBasis(int dim_rho_, LindbladType lindbladtype_) : GellmannBasis(dim_rho_, false, lindbladtype_) {

  /* Assemble system Matrices */
  assembleSystemMats();

  /* Allocate assembled operator */
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Re);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Im);
  MatSetUp(Operator_Re);
  MatSetUp(Operator_Im);
  MatAssemblyBegin(Operator_Re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Operator_Im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Operator_Re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Operator_Im, MAT_FINAL_ASSEMBLY);
}

HamiltonianBasis::~HamiltonianBasis(){
  MatDestroy(&Operator_Re);
  MatDestroy(&Operator_Im);
}

void HamiltonianBasis::assembleSystemMats(){

  /* Set up and store the Hamiltonian system matrices:
   *   (-i*sigma)   or vectorized   -i(I kron sigma - sigma kron I) 
   *  A = Re(-isigma)
   *  B = Im(-isigma)
   */

  //if vectorizing, set up the identity matrix
  Mat Id; 
  if (lindbladtype != LindbladType::NONE) {
    MatCreate(PETSC_COMM_WORLD, &Id);
    MatSetType(Id, MATSEQAIJ);
    MatSetSizes(Id, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    for (int i=0; i<dim_rho; i++){
      MatSetValue(Id, i, i, 1.0, INSERT_VALUES);
    }
    MatAssemblyBegin(Id, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Id, MAT_FINAL_ASSEMBLY);
  }

  // Set up -i*(Real_Gellmann), they go into Bd = Im(-iH)
  for (int i=0; i<BasisMat_Re.size(); i++){
    Mat myMat;
    if (lindbladtype == LindbladType::NONE){
      MatDuplicate(BasisMat_Re[i],  MAT_COPY_VALUES, &myMat);
      MatScale(myMat, -1.0);
    } else {
      Mat myMat1;
      MatSeqAIJKron(BasisMat_Re[i], Id, MAT_INITIAL_MATRIX, &myMat);  // sigma^T kron I
      MatSeqAIJKron(Id, BasisMat_Re[i], MAT_INITIAL_MATRIX, &myMat1); // I kron sigma
      MatAXPY(myMat, -1.0, myMat1, DIFFERENT_NONZERO_PATTERN);
      MatDestroy(&myMat1);
    }
    SystemMats_B.push_back(myMat);
  }

  // Set up -i*(Imag_BasisMat), they go into Ad = Re(-iH) [note: no scaling by -1!]
  Mat myMat;
  for (int i=0; i<BasisMat_Im.size(); i++){
    if (lindbladtype == LindbladType::NONE){
      MatDuplicate(BasisMat_Im[i],  MAT_COPY_VALUES, &myMat);
    } else {
      Mat myMat1;
      MatSeqAIJKron(BasisMat_Im[i], Id, MAT_INITIAL_MATRIX, &myMat);  // sigma^T kron I
      MatSeqAIJKron(Id, BasisMat_Im[i], MAT_INITIAL_MATRIX, &myMat1); // I kron sigma
      MatAXPY(myMat, 1.0, myMat1, DIFFERENT_NONZERO_PATTERN);
      MatDestroy(&myMat1);
    }
    SystemMats_A.push_back(myMat);
  }
  if (lindbladtype != LindbladType::NONE) {
    MatDestroy(&Id);
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

void HamiltonianBasis::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, int skipID){
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

void HamiltonianBasis::assembleOperator(std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im){

  /* Assemble the Hamiltonian */
  for (int i=0; i<BasisMat_Re.size(); i++) {
    MatAXPY(Operator_Re, learnparams_Re[i] / (2.0*M_PI), BasisMat_Re[i], DIFFERENT_NONZERO_PATTERN);
  }
  for (int i=0; i<BasisMat_Im.size(); i++) {
    MatAXPY(Operator_Im, learnparams_Im[i] / (2.0*M_PI), BasisMat_Im[i], DIFFERENT_NONZERO_PATTERN);
  }
}


LindbladBasis::LindbladBasis(int dim_rho_) : GellmannBasis(dim_rho_, true, LindbladType::BOTH) {

  /* Assemble system Matrices */
  assembleSystemMats();
}

LindbladBasis::~LindbladBasis(){}


void LindbladBasis::assembleSystemMats(){
  /* Set up and store the Lindblad system matrices: 
  *   sigma.conj kron sigma - 1/2(I kron sigma^t sigma + (sigma^t sigma)^T kron I)
   * Here, all Basis mats are REAL (only using upper part of the real Gellmann mats), hence all go into A = Re(...)
   * Note that here we have: sigma.conj = sigma and (sigma^tsigma)^T
  */

  // Set up the identity matrix
  Mat Id;
  MatCreate(PETSC_COMM_WORLD, &Id);
  MatSetType(Id, MATSEQAIJ);
  MatSetSizes(Id, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
  for (int i=0; i<dim_rho; i++){
    MatSetValue(Id, i, i, 1.0, INSERT_VALUES);
  }
  MatAssemblyBegin(Id, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Id, MAT_FINAL_ASSEMBLY);

  for (int i=0; i<BasisMat_Re.size(); i++){

    Mat myMat, myMat1, myMat2, sigmasq;
    MatTransposeMatMult(BasisMat_Re[i], BasisMat_Re[i],MAT_INITIAL_MATRIX, PETSC_DEFAULT,&sigmasq);

    MatSeqAIJKron(BasisMat_Re[i], BasisMat_Re[i], MAT_INITIAL_MATRIX, &myMat);   // sigma \kron sigma
    MatSeqAIJKron(Id, sigmasq, MAT_INITIAL_MATRIX, &myMat1);  // Id \kron sigma^tsigma
    MatAXPY(myMat, -0.5, myMat1, DIFFERENT_NONZERO_PATTERN);
    MatSeqAIJKron(sigmasq, Id, MAT_INITIAL_MATRIX, &myMat2);  // sigma^tsigma \kron Id
    MatAXPY(myMat, -0.5, myMat2, DIFFERENT_NONZERO_PATTERN);

    SystemMats_A.push_back(myMat);
    MatDestroy(&myMat1);
    MatDestroy(&myMat2);
    MatDestroy(&sigmasq);
  }
  MatDestroy(&Id);
}

void LindbladBasis::applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL_Re, std::vector<double>& learnparamsL_Im){

  // Real parts of lindblad terms
  for (int i=0; i< SystemMats_A.size(); i++){
    // uout += learnparam_Re * SystemA * u
    MatMult(SystemMats_A[i], u, aux);
    VecAXPY(uout, learnparamsL_Re[i], aux); 
    // vout += learnparam_Re * SystemA * v
    MatMult(SystemMats_A[i], v, aux);
    VecAXPY(vout, learnparamsL_Re[i], aux);
  }
  // Imaginary parts of lindblad terms
  for (int i=0; i< SystemMats_B.size(); i++){
    // uout -= learnparam_Im * SystemB * v
    MatMult(SystemMats_B[i], v, aux);
    VecAXPY(uout, -1.*learnparamsL_Im[i], aux); 
    // vout += learnparam_Im * SystemB * u
    MatMult(SystemMats_B[i], u, aux);
    VecAXPY(vout, learnparamsL_Im[i], aux);
  }
}

void LindbladBasis::applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparamsL_Re, std::vector<double>& learnparamsL_Im){
  // Real parts of lindblad terms
  for (int i=0; i< SystemMats_A.size(); i++){
    // uout += learnparam_Re * SystemMat_A^T * u
    MatMultTranspose(SystemMats_A[i], u, aux);
    VecAXPY(uout, learnparamsL_Re[i], aux); 
    // vout += learnparam_Re * SystemMat_A^T * v
    MatMultTranspose(SystemMats_A[i], v, aux);
    VecAXPY(vout, learnparamsL_Re[i], aux);
  }
  // Imaginary parts of lindbladterms
  for (int i=0; i< SystemMats_B.size(); i++){
    // uout += learnparam_Im * SystemMat_B^T * v
    MatMultTranspose(SystemMats_B[i], v, aux);
    VecAXPY(uout, learnparamsL_Im[i], aux); 
    // vout -= learnparam_Im * SystemMat_B^T * u
    MatMultTranspose(SystemMats_B[i], u, aux);
    VecAXPY(vout, -1.*learnparamsL_Im[i], aux);
  }
}


void LindbladBasis::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, int skip){
 double uAubar, vAvbar, vBubar, uBvbar;
  // Note: Storage in grad corresponds to x = [learn_H, learn_L], so need to skip to the second part of the gradient] 

  for (int i=0; i< SystemMats_A.size(); i++){
    MatMult(SystemMats_A[i], u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(SystemMats_A[i], v, aux); VecDot(aux, vbar, &vAvbar);
    VecSetValue(grad, i+skip, alpha*(uAubar + vAvbar), ADD_VALUES);
  }
  skip += SystemMats_A.size();
  for (int i=0; i<SystemMats_B.size(); i++){
    MatMult(SystemMats_B[i], u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(SystemMats_B[i], v, aux); VecDot(aux, ubar, &vBubar);
    VecSetValue(grad, i+skip, alpha*(-vBubar + uBvbar), ADD_VALUES);
  }
}



Learning::Learning(std::vector<int>& nlevels, LindbladType lindbladtype_, std::vector<std::string>& learninit_str, std::string data_name, double data_dtAWG_, int data_ntime_, int loss_every_k_, std::default_random_engine rand_engine, bool quietmode_){
  lindbladtype = lindbladtype_;
  data_dtAWG = data_dtAWG_;
  data_ntime = data_ntime_;
  quietmode = quietmode_;
  loss_every_k = loss_every_k_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  loss_integral = 0.0;

  // Get dimension of the Hilbert space (dim_rho = N) and dimension of the state variable (dim = N or N^2 for Schroedinger or Lindblad solver)
  dim_rho = 1;
  for (int i=0; i<nlevels.size(); i++){
    dim_rho *= nlevels[i];
  }
  dim = dim_rho;
  if (lindbladtype != LindbladType::NONE){
    dim = dim_rho*dim_rho; 
  }


  /* Proceed only if this is not a dummy class (aka only if using the UDE model)*/
  if (dim_rho > 0) {

    /* Create Basis for the learnable terms. Here: generalized Gellman matrices */
    hamiltonian_basis = new HamiltonianBasis(dim_rho, lindbladtype);
    if (lindbladtype != LindbladType::NONE) {
      lindblad_basis    = new LindbladBasis(dim_rho); 
    } else {
      lindblad_basis    = new LindbladBasis(0);  // will be empty if not Lindblad solver
    }

    /* Set the total number of learnable paramters */
    nparams = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis();

    /* Allocate learnable Hamiltonian and Lindblad parameters, and set an initial guess */
    initLearnParams(learninit_str, rand_engine);

    /* Load trajectory data from file */
    loadData(data_name, data_dtAWG, data_ntime);

    /* Create auxiliary vectors needed for MatMult. */
    VecCreate(PETSC_COMM_WORLD, &aux2);    // aux2 sized for state (re and im)
    VecSetSizes(aux2, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(aux2);

    // Some output 
    if (!quietmode) {
      printf("Learning with %d Gellmann mats\n", hamiltonian_basis->getNBasis());
    }
  }
}

Learning::~Learning(){
  if (dim_rho > 0) {
    learnparamsH_Re.clear();
    learnparamsH_Im.clear();
    for (int i=0; i<data.size(); i++){
      VecDestroy(&data[i]);
    }
    data.clear();
    VecDestroy(&aux2);

    delete hamiltonian_basis;
    delete lindblad_basis;
  }
}

void Learning::applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout){

  if (dim_rho <= 0) return;

  hamiltonian_basis->applySystem(u, v, uout, vout, learnparamsH_Re, learnparamsH_Im);
  lindblad_basis->applySystem(u, v, uout, vout, learnparamsL_Re, learnparamsL_Im);
}




void Learning::applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout){

  if (dim_rho <= 0) return;

  hamiltonian_basis->applySystem_diff(u,v,uout, vout, learnparamsH_Re, learnparamsH_Im);
  lindblad_basis->applySystem_diff(u,v,uout, vout, learnparamsL_Re, learnparamsL_Im);
}


void Learning::viewOperators(){

  if (dim_rho <= 0) return;

  hamiltonian_basis->assembleOperator(learnparamsH_Re, learnparamsH_Im);
  printf("\nLearned Hamiltonian operator: Re = \n");
  MatView(hamiltonian_basis->getOperator_Re(), NULL);
  printf("Learned Hamiltonian operator: Im = \n");
  MatView(hamiltonian_basis->getOperator_Im(), NULL);

  for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
    printf("Lindblad: %d \n", i);
    MatScale(lindblad_basis->getBasisMat_Re(i), learnparamsL_Re[i]);
    MatView(lindblad_basis->getBasisMat_Re(i), NULL);
    // Revert scaling, just to be safe...
    MatScale(lindblad_basis->getBasisMat_Re(i), 1.0/learnparamsL_Re[i]);
  }
}


void Learning::setLearnParams(const Vec x){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
   */

  const PetscScalar* ptr;
  VecGetArrayRead(x, &ptr);
  
  // Hamiltonian parameters first
  for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++) {
    learnparamsH_Re[i] = ptr[i];
  }
  int skip = hamiltonian_basis->getNBasis_Re();
  for (int i=0; i<hamiltonian_basis->getNBasis_Im(); i++){
    learnparamsH_Im[i] = ptr[i+skip];
  }
  // Lindblad terms next
  skip = hamiltonian_basis->getNBasis();
  for (int i=0; i<lindblad_basis->getNBasis_Re(); i++) {
    learnparamsL_Re[i] = ptr[i+skip];
  }
  skip += lindblad_basis->getNBasis_Re();
  for (int i=0; i<lindblad_basis->getNBasis_Im(); i++){
    learnparamsL_Im[i] = ptr[i+skip];
  }

  VecRestoreArrayRead(x, &ptr);
}

void Learning::getLearnParams(double* x){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
   */
  // Hamiltonian parameters first
  for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++) {
    x[i]      = learnparamsH_Re[i];
  }
  int skip = hamiltonian_basis->getNBasis_Re();
  for (int i=0; i<hamiltonian_basis->getNBasis_Im(); i++){
    x[i+skip] = learnparamsH_Im[i];
  }
  // Lindblad terms next
  skip = hamiltonian_basis->getNBasis();
  for (int i=0; i<lindblad_basis->getNBasis_Re(); i++) {
    x[i+skip] = learnparamsL_Re[i];
  }
  skip += lindblad_basis->getNBasis_Re();
  for (int i=0; i<lindblad_basis->getNBasis_Im(); i++){
    x[i+skip] = learnparamsL_Im[i];
  }
}

void Learning::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
   */

  if (dim_rho <= 0) return;

  hamiltonian_basis->dRHSdp(grad, u, v, alpha, ubar, vbar);
  lindblad_basis->dRHSdp(grad, u, v, alpha, ubar, vbar, hamiltonian_basis->getNBasis());

  VecAssemblyBegin(grad);
  VecAssemblyEnd(grad);
}


void Learning::loadData(std::string data_name, double data_dtAWG, int data_ntime){

  // Open files 
  std::ifstream infile_re;
  std::ifstream infile_im;
  infile_re.open(data_name + "_re.dat", std::ifstream::in);
  infile_im.open(data_name + "_im.dat", std::ifstream::in);
  if(infile_re.fail() || infile_im.fail() ) {// checks to see if file opended 
      std::cout << "\n ERROR loading learning data file\n" << std::endl; 
      std::cout << data_name + "_re.dat" << std::endl;
      exit(1);
  } else {
    if (!quietmode) {
      std::cout<< "Loading trajectory data from " << data_name+"_re.dat" << ", " << data_name+"_im.dat" << std::endl;
    }
  }

  // Iterate over each line in the file
  for (int n = 0; n <data_ntime; n++) {
    Vec state;
    VecCreate(PETSC_COMM_WORLD, &state);
    VecSetSizes(state, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(state);

    // Iterate over columns
    double val_re, val_im;
    infile_re >> val_re; // first element is time
    infile_im >> val_im; // first element is time
    // printf("time-step %1.4f == %1.4f ??\n", val_re, val_im);
    assert(fabs(val_re - val_im) < 1e-12);
    for (int i=0; i<dim; i++) { // Other elements are the state (re and im) at this time
      infile_re >> val_re;
      infile_im >> val_im;
      VecSetValue(state, getIndexReal(i), val_re, INSERT_VALUES);
      VecSetValue(state, getIndexImag(i), val_im, INSERT_VALUES);
    }
    VecAssemblyBegin(state);
    VecAssemblyEnd(state);
    data.push_back(state);
  }

  // Close files
	infile_re.close();
	infile_im.close();

  // // TEST what was loaded
  // printf("\nDATA POINTS:\n");
  // for (int i=0; i<data.size(); i++){
  //   VecView(data[i], NULL);
  // }
  // printf("END DATA POINTS.\n\n");
}



void Learning::initLearnParams(std::vector<std::string> learninit_str, std::default_random_engine rand_engine){
  // Switch over initialization string ("file", "constant", or "random")

  if (learninit_str[0].compare("file") == 0 ) { //  Read parameter from file. 

    /* Parameter file format:  One column containing all learnable paramters as 
     *    x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
     */
    int nparams = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis();
    std::vector<double> initguess_fromfile(nparams, 0.0);
    assert(learninit_str.size()>1);
    if (mpirank_world == 0) {
      read_vector(learninit_str[1].c_str(), initguess_fromfile.data(), nparams, quietmode);
    }
    MPI_Bcast(initguess_fromfile.data(), nparams, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
    //First set all Hamiltonian parameters
    for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++){
      learnparamsH_Re.push_back(initguess_fromfile[i]); 
    }
    int skip = hamiltonian_basis->getNBasis_Re();
    for (int i=0; i<=hamiltonian_basis->getNBasis_Im(); i++){
      learnparamsH_Im.push_back(initguess_fromfile[i + skip]); 
    }
    // Then set all Lindblad params
    skip = hamiltonian_basis->getNBasis();
    for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
      learnparamsL_Re.push_back(initguess_fromfile[skip + i]); 
    }
    skip = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis_Re();
    for (int i=0; i<=lindblad_basis->getNBasis_Im(); i++){
      learnparamsL_Im.push_back(initguess_fromfile[skip + i]); 
    }
  } else if (learninit_str[0].compare("random") == 0 ) {
    // Set uniform random parameters in [0,amp)

    // First all Hamiltonian parameters, multiply by 2*M_PI
    assert(learninit_str.size()>1);
    double amp = atof(learninit_str[1].c_str());
    std::uniform_real_distribution<double> unit_dist(0.0, amp);
    for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++){
      learnparamsH_Re.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
    }
    for (int i=0; i<hamiltonian_basis->getNBasis_Im(); i++){
      learnparamsH_Im.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
    }
    // Then all Lindblad parameters
    if (lindblad_basis->getNBasis() > 0) {
      if (learninit_str.size() == 2) learninit_str.push_back(learninit_str[1]);
      assert(learninit_str.size()>2);
      amp = atof(learninit_str[2].c_str());
      std::uniform_real_distribution<double> unit_dist2(0.0, amp);
      for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
        learnparamsL_Re.push_back(unit_dist2(rand_engine)); // ns?
      }
      for (int i=0; i<lindblad_basis->getNBasis_Im(); i++){
        learnparamsL_Im.push_back(unit_dist2(rand_engine)); // ns? 
      }
    }
  } else if (learninit_str[0].compare("constant") == 0 ) {
    // Set constant amp
    // First all Hamiltonian parameters
    assert(learninit_str.size()>1);
    double amp = atof(learninit_str[1].c_str());
    for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++){
      learnparamsH_Re.push_back(amp * 2.0*M_PI);
    }
    for (int i=0; i<hamiltonian_basis->getNBasis_Im(); i++){
      learnparamsH_Im.push_back(amp * 2.0*M_PI);
    }
    // Then all Lindblad parameters
    if (lindblad_basis->getNBasis() > 0) {
      if (learninit_str.size() == 2) learninit_str.push_back(learninit_str[1]);
      assert(learninit_str.size()>2);
      amp = atof(learninit_str[2].c_str());
      for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
        learnparamsL_Re.push_back(amp); // ns?
      }
      for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
        learnparamsL_Im.push_back(amp); // ns? 
      }
    }
  } else {
    printf("ERROR: Wrong configuration for learnable parameter initialization. Choose 'file, <pathtofile>', or 'random, <amplitude_Ham>, <amplitude_Lindblad>', or 'constant, <amplitude_Ham>, <amplitude_Lind>'\n");
    exit(1);
  }
}


void Learning::addToLoss(int timestepID, Vec x){

  if (dim_rho <= 0) return;

  // Add to loss only every k-th timestep, and if data exists
  int dataID = -1;
  if (timestepID % loss_every_k == 0 ) {
    dataID = timestepID / loss_every_k;
  }

  // Add to loss if data exists
  if (dataID > 0 && dataID < getNData()) {
    // printf("Add to loss at ts %d with dataID %d\n", timestepID, dataID);
    double norm; 
    Vec xdata = getData(dataID);
    // Frobenius norm between state x and data
    VecAYPX(aux2, 0.0, x);
    VecAXPY(aux2, -1.0, xdata);   // aux2 = x - data
    VecNorm(aux2, NORM_2, &norm);
    loss_integral += 0.5*norm*norm / (getNData()-1);
  }
}


void Learning::addToLoss_diff(int timestepID, Vec xbar, Vec xprimal, double Jbar_loss){

  if (dim_rho <= 0) return;

  // Add to loss only every k-th timestep, and if data exists
  int dataID = -1;
  if (timestepID % loss_every_k == 0 ) {
    dataID = timestepID / loss_every_k;
  }

  if (dataID > 0 && dataID < getNData()) {
    // printf("loss_DIFF at ts %d with dataID %d\n", timestepID, dataID);
    Vec xdata = getData(dataID);
    VecAXPY(xbar, Jbar_loss / (getNData()-1), xprimal);
    VecAXPY(xbar, -Jbar_loss/ (getNData()-1), xdata);
  }
}


