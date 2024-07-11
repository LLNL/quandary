#include "learning.hpp"

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

    /* Create Basis for the learnable Hamiltonian terms. Here: generalized Gellman matrices */
    bool vectorize = false;
    if (lindbladtype != LindbladType::NONE) vectorize = true;
    hamiltonian_basis = new HamiltonianBasis(dim_rho, vectorize);

    // /* Create Basis for the learnable Lindblad terms. Here: upper triangular part of the generalized Gellman matrices */
    // if (lindbladtype != LindbladType::NONE) {
    //   lindblad_basis = new LindbladBasis(dim_rho);
    // } else {
    //   lindblad_basis = new LindbladBasis(0); // DUMMY! Empty basis.
    // }

    /* Set the total number of learnable paramters */
    // nparams = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis();
    nparams = hamiltonian_basis->getNBasis(); 

    /* Allocate learnable Hamiltonian and Lindblad parameters, and set an initial guess */
    initLearnableParams(learninit_str, rand_engine);

    /* Load trajectory data from file */
    loadData(data_name, data_dtAWG, data_ntime);

    /* Create auxiliary vectors needed for MatMult. */
    VecCreate(PETSC_COMM_WORLD, &aux);     // aux sized for Re(state) or Im(state) 
    VecCreate(PETSC_COMM_WORLD, &aux2);    // aux2 sized for state (re and im)
    VecSetSizes(aux , PETSC_DECIDE, dim);
    VecSetSizes(aux2, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(aux);
    VecSetFromOptions(aux2);

    // Some output 
    if (!quietmode) {
      printf("Learning with %d Gellmann mats\n", hamiltonian_basis->getNBasis());
    }
  }
}

Learning::~Learning(){
  if (dim_rho > 0) {
    learnparamsH_A.clear();
    learnparamsH_B.clear();
    for (int i=0; i<data.size(); i++){
      VecDestroy(&data[i]);
    }
    data.clear();
    VecDestroy(&aux);
    VecDestroy(&aux2);

    delete hamiltonian_basis;
    // delete lindblad_basis;
  }
}

void Learning::applyLearnHamiltonian(Vec u, Vec v, Vec uout, Vec vout){

  // Real parts of (-i * H)
  for (int i=0; i< hamiltonian_basis->getNBasis_A(); i++){
    // uout += learnparamA * GellmannA * u
    MatMult(hamiltonian_basis->getSystemMat_A(i), u, aux);
    VecAXPY(uout, learnparamsH_A[i], aux); 
    // vout += learnparamA * GellmannA * v
    MatMult(hamiltonian_basis->getSystemMat_A(i), v, aux);
    VecAXPY(vout, learnparamsH_A[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< hamiltonian_basis->getNBasis_B(); i++){
    // uout -= learnparamB * GellmannB * v
    MatMult(hamiltonian_basis->getSystemMat_B(i), v, aux);
    VecAXPY(uout, -1.*learnparamsH_B[i], aux); 
    // vout += learnparamB * GellmannB * u
    MatMult(hamiltonian_basis->getSystemMat_B(i), u, aux);
    VecAXPY(vout, learnparamsH_B[i], aux);
  }
}


void Learning::applyLearnLindblad(Vec u, Vec v, Vec uout, Vec vout){
  printf("TODO\n");
  exit(1);
}

void Learning::applyLearnLindblad_diff(Vec u, Vec v, Vec uout, Vec vout){
  printf("TODO\n");
  exit(1);
}

void Learning::applyLearnHamiltonian_diff(Vec u, Vec v, Vec uout, Vec vout){

  // Real parts of (-i * H)
  for (int i=0; i< hamiltonian_basis->getNBasis_A(); i++){
    // uout += learnparamA * GellmannA^T * u
    MatMultTranspose(hamiltonian_basis->getSystemMat_A(i), u, aux);
    VecAXPY(uout, learnparamsH_A[i], aux); 
    // vout += learnparamA * GellmannA^T * v
    MatMultTranspose(hamiltonian_basis->getSystemMat_A(i), v, aux);
    VecAXPY(vout, learnparamsH_A[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< hamiltonian_basis->getNBasis_B(); i++){
    // uout += learnparamB * GellmannB^T * v
    MatMultTranspose(hamiltonian_basis->getSystemMat_B(i), v, aux);
    VecAXPY(uout, learnparamsH_B[i], aux); 
    // vout -= learnparamB * GellmannB^T * u
    MatMultTranspose(hamiltonian_basis->getSystemMat_B(i), u, aux);
    VecAXPY(vout, -1.*learnparamsH_B[i], aux);
  }
}


void Learning::assembleHamiltonian(Mat& Re, Mat& Im){

  MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Re);
  MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Im);
  MatSetUp(Re);
  MatSetUp(Im);
  MatAssemblyBegin(Re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Im, MAT_FINAL_ASSEMBLY);

  /* Assemble the Hamiltonian */
  // Note, the Gellmann BasisMats store A=Re(-i*sigma) and B=Im(-i*sigma), here we want to return A=sum Re(sigma) and B=sum Im(sigma), hence need to revert order (learnparams_A are for GellmannBasis_Re)

  for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++) {
    MatAXPY(Re, learnparamsH_B[i] / (2.0*M_PI), hamiltonian_basis->getBasisMat_Re(i), DIFFERENT_NONZERO_PATTERN);
  }
  for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++) {
    MatAXPY(Im, learnparamsH_A[i] / (2.0*M_PI), hamiltonian_basis->getBasisMat_Im(i), DIFFERENT_NONZERO_PATTERN);
  }
}


void Learning::setLearnParams(const Vec x){

  /* Storage of parameters in x: First all for SystemMats_A, then all for SystemMats_B */

  const PetscScalar* ptr;
  VecGetArrayRead(x, &ptr);
  for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++) {
    learnparamsH_A[i] = ptr[i];
  }
  int skip = hamiltonian_basis->getNBasis_A();
  for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++){
    learnparamsH_B[i] = ptr[i+skip];
  }
  VecRestoreArrayRead(x, &ptr);
}

void Learning::getLearnParams(double* x){
  for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++) {
    x[i]        = learnparamsH_A[i];
  }
  int skip = hamiltonian_basis->getNBasis_A();
  for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++){
    x[i+skip] = learnparamsH_B[i];
  }
}

void Learning::dRHSdp_Ham(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar){
  double uAubar, vAvbar, vBubar, uBvbar;

  // gamma_bar_A += alpha * (  u^t sigma_A^t ubar + v^t sigma_A^t vbar )
  // gamma_bar_B += alpha * ( -v^t sigma_B^t ubar + u^t sigma_B^t vbar )

  for (int i=0; i< hamiltonian_basis->getNBasis_A(); i++){

    MatMult(hamiltonian_basis->getSystemMat_A(i), u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(hamiltonian_basis->getSystemMat_A(i), v, aux); VecDot(aux, vbar, &vAvbar);
    VecSetValue(grad, i, alpha*(uAubar + vAvbar), ADD_VALUES);
  }
  int skip = hamiltonian_basis->getNBasis_A();
  for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++){
    MatMult(hamiltonian_basis->getSystemMat_B(i), u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(hamiltonian_basis->getSystemMat_B(i), v, aux); VecDot(aux, ubar, &vBubar);
    VecSetValue(grad, i+skip, alpha*(-vBubar + uBvbar), ADD_VALUES);
  }

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



void Learning::initLearnableParams(std::vector<std::string> learninit_str, std::default_random_engine rand_engine){

  // Switch over initialization string ("file", "constant", or "random")
  if (learninit_str[0].compare("file") == 0 ) {
    // Read parameter from file. 

    /* Parameter file format: First all that corresponds to SystemMats_A = Real(-i*sigma), hence all those that correspond to purely imaginary basis matrices (for which -i*sigma) is real. Then all SystemMats_B = Imag(-i*sigma) where first come all offdiagonal ones, then all diagonal ones. */

    assert(learninit_str.size()>1);
    std::vector<double> initguess_fromfile(nparams, 0.0);
    if (mpirank_world == 0) {
      read_vector(learninit_str[1].c_str(), initguess_fromfile.data(), nparams, quietmode);
    }
    MPI_Bcast(initguess_fromfile.data(), nparams, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
    //First set all Hamiltonian parameters
    for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++){
      learnparamsH_A.push_back(initguess_fromfile[i]); 
    }
    for (int i=0; i<=hamiltonian_basis->getNBasis_B(); i++){
      learnparamsH_B.push_back(initguess_fromfile[i + hamiltonian_basis->getNBasis_A()]); 
    }
    int skip = hamiltonian_basis->getNBasis();

    // // Then set all Lindblad params
    // for (int i=0; i<lindblad_basis->getNBasis_A(); i++){
    //   learnparamsL_A.push_back(initguess_fromfile[skip + i]); 
    // }
    // for (int i=0; i<=lindblad_basis->getNBasis_B(); i++){
    //   learnparamsL_B.push_back(initguess_fromfile[skip + i + lindblad_basis->getNBasis_A()]); 
    // }
  } else if (learninit_str[0].compare("random") == 0 ) {
    // Set uniform random parameters in [0,amp)

    // First all Hamiltonian parameters
    assert(learninit_str.size()>1);
    double amp = atof(learninit_str[1].c_str());
    std::uniform_real_distribution<double> unit_dist(0.0, amp);
    for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++){
      learnparamsH_A.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
    }
    for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++){
      learnparamsH_B.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
    }
    // // Then all Lindblad parameters
    // assert(learninit_str.size()>2);
    // amp = atof(learninit_str[2].c_str());
    // std::uniform_real_distribution<double> unit_dist(0.0, amp);
    // for (int i=0; i<lindblad_basis->getNBasis_A(); i++){
    //   learnparamsL_A.push_back(unit_dist(rand_engine)); // ns?
    // }
    // for (int i=0; i<lindblad_basis->getNBasis_B(); i++){
    //   learnparamsL_B.push_back(unit_dist(rand_engine)); // ns? 
    // }

  } else if (learninit_str[0].compare("constant") == 0 ) {
    // Set constant amp
    // First all Hamiltonian parameters
    assert(learninit_str.size()>1);
    double amp = atof(learninit_str[1].c_str());
    for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++){
      learnparamsH_A.push_back(amp * 2.0*M_PI);
    }
    for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++){
      learnparamsH_B.push_back(amp * 2.0*M_PI);
    }
    // // Then all Lindblad parameters
    // assert(learninit_str.size()>2);
    // amp = atof(learninit_str[2].c_str());
    // for (int i=0; i<lindblad_basis->getNBasis_A(); i++){
    //   learnparamsL_A.push_back(amp); // ns?
    // }
    // for (int i=0; i<lindblad_basis->getNBasis_B(); i++){
    //   learnparamsL_B.push_back(amp); // ns? 
    // }
  } else {
    printf("ERROR: Wrong configuration for learnable parameter initialization. Choose 'file, <pathtofile>', or 'random, <amplitude_Ham>, <amplitude_Lindblad>', or 'constant, <amplitude_Ham>, <amplitude_Lind>'\n");
    exit(1);
  }
}


void Learning::addToLoss(int timestepID, Vec x){

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


GellmannBasis::GellmannBasis(int dim_rho_, bool upper_only_, bool vectorize_){
  dim_rho = dim_rho_;
  dim = dim_rho;
  vectorize = vectorize_;
  if (vectorize){
    dim = dim_rho*dim_rho; 
  }
  upper_only = upper_only_;


  /* First all offdiagonal matrices (re and im)*/
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      Mat G_re, G_im;
      MatCreate(PETSC_COMM_WORLD, &G_re);
      MatCreate(PETSC_COMM_WORLD, &G_im);
      MatSetType(G_re, MATSEQAIJ);
      MatSetType(G_im, MATSEQAIJ);
      MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
      MatSetSizes(G_im, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
      MatSetUp(G_re);
      MatSetUp(G_im);

      /* Real sigma_jk^re = |j><k| + |k><j| */ 
      MatSetValue(G_re, j, k, 1.0, INSERT_VALUES);
      if (!upper_only) MatSetValue(G_re, k, j, 1.0, INSERT_VALUES);
      
      /* Imaginary sigma_jk^im = -i|j><k| + i|k><j| */ 
      MatSetValue(G_im, j, k, -1.0, INSERT_VALUES);
      if (!upper_only) MatSetValue(G_im, k, j, +1.0, INSERT_VALUES);

      /* Assemble and store */
      MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyBegin(G_im, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(G_im, MAT_FINAL_ASSEMBLY);
      BasisMat_Re.push_back(G_re);
      BasisMat_Im.push_back(G_im);
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
}


HamiltonianBasis::HamiltonianBasis(int dim_rho_, bool vectorize_) : GellmannBasis(dim_rho_, false, vectorize_) {

  /* Set up and store the Hamiltonian system matrices:
   *   (-i*sigma)   or vectorized   -i(I kron sigma - sigma kron I) 
   */

  //if vectorizing, set up the identity matrix
  Mat Id; 
  if (vectorize) {
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
    if (!vectorize){
      MatDuplicate(BasisMat_Re[i],  MAT_COPY_VALUES, &myMat);
      MatScale(myMat, -1.0);
    } else {
      Mat myMat1;
      MatSeqAIJKron(BasisMat_Re[i], Id, MAT_INITIAL_MATRIX, &myMat);  // sigma^T kron I
      MatSeqAIJKron(Id, BasisMat_Re[i], MAT_INITIAL_MATRIX, &myMat1); // I kron sigma
      MatAXPY(myMat, -1.0, myMat1, DIFFERENT_NONZERO_PATTERN);
    }
    SystemMats_B.push_back(myMat);
  }

  // Set up -i*(Imag_BasisMat), they go into Ad = Re(-iH) [note: no scaling by -1!]
  Mat myMat;
  for (int i=0; i<BasisMat_Im.size(); i++){
    if (!vectorize){
      MatDuplicate(BasisMat_Im[i],  MAT_COPY_VALUES, &myMat);
    } else {
      Mat myMat1;
      MatSeqAIJKron(BasisMat_Im[i], Id, MAT_INITIAL_MATRIX, &myMat);  // sigma^T kron I
      MatSeqAIJKron(Id, BasisMat_Im[i], MAT_INITIAL_MATRIX, &myMat1); // I kron sigma
      MatAXPY(myMat, 1.0, myMat1, DIFFERENT_NONZERO_PATTERN);
    }
    SystemMats_A.push_back(myMat);
  }
  if (vectorize) MatDestroy(&Id);
}


HamiltonianBasis::~HamiltonianBasis(){
  for (int i=0; i< SystemMats_A.size(); i++){
    MatDestroy(&SystemMats_A[i]);
  }
  for (int i=0; i<SystemMats_B.size(); i++){
    MatDestroy(&SystemMats_B[i]);
  }
  SystemMats_A.clear();
  SystemMats_B.clear();
 
}


  
