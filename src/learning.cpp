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
    MatMult(hamiltonian_basis->getMat_A(i), u, aux);
    VecAXPY(uout, learnparamsH_A[i], aux); 
    // vout += learnparamA * GellmannA * v
    MatMult(hamiltonian_basis->getMat_A(i), v, aux);
    VecAXPY(vout, learnparamsH_A[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< hamiltonian_basis->getNBasis_B(); i++){
    // uout -= learnparamB * GellmannB * v
    MatMult(hamiltonian_basis->getMat_B(i), v, aux);
    VecAXPY(uout, -1.*learnparamsH_B[i], aux); 
    // vout += learnparamB * GellmannB * u
    MatMult(hamiltonian_basis->getMat_B(i), u, aux);
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
    MatMultTranspose(hamiltonian_basis->getMat_A(i), u, aux);
    VecAXPY(uout, learnparamsH_A[i], aux); 
    // vout += learnparamA * GellmannA^T * v
    MatMultTranspose(hamiltonian_basis->getMat_A(i), v, aux);
    VecAXPY(vout, learnparamsH_A[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< hamiltonian_basis->getNBasis_B(); i++){
    // uout += learnparamB * GellmannB^T * v
    MatMultTranspose(hamiltonian_basis->getMat_B(i), v, aux);
    VecAXPY(uout, learnparamsH_B[i], aux); 
    // vout -= learnparamB * GellmannB^T * u
    MatMultTranspose(hamiltonian_basis->getMat_B(i), u, aux);
    VecAXPY(vout, -1.*learnparamsH_B[i], aux);
  }
}


void Learning::getHamiltonian(Mat& Re, Mat& Im){

  MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Re);
  MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Im);
  MatSetUp(Re);
  MatSetUp(Im);
  MatAssemblyBegin(Re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Im, MAT_FINAL_ASSEMBLY);

  if (dim > dim_rho) {
    printf("getHamiltonianOperator() not implemented for Lindblad solver currently. Sorry!\n");
    exit(1);
  }

  /* Assemble the Hamiltonian */
  // Note, the Gellmann BasisMats store A=Re(-i*sigma) and B=Im(-i*sigma), here we want to return A=sum Re(sigma) and B=sum Im(sigma), hence need to revert order.

  // -> if sigma was purely real, then -i*sigma is purely imaginary and hence stored in BasisMats_B
  for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++) {
    double fac = -learnparamsH_B[i] / (2.0*M_PI);
    MatAXPY(Re, fac , hamiltonian_basis->getMat_B(i), SUBSET_NONZERO_PATTERN);
  }
  // -> if sigma was purely imaginary, then -i*sigma is purely real and hence stored in BasisMats_A. Note that the -1 cancels out??
  for (int i=0; i<hamiltonian_basis->getNBasis_A(); i++) {
    double fac = learnparamsH_A[i] / (2.0*M_PI);
    MatAXPY(Im, fac, hamiltonian_basis->getMat_A(i), SUBSET_NONZERO_PATTERN);
  }
}


void Learning::setLearnParams(const Vec x){

  /* Storage of parameters in x: First all for BasisMats_A, then all for BasisMats_B */

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

  // for (int i=0; i<learnparamsH_A.size(); i++){
  //   printf("A %1.14e% \n", learnparamsH_A[i]);
  // }
  // for (int i=0; i<learnparamsH_B.size(); i++){
  //   printf("B %1.14e \n", learnparamsH_B[i]);
  // }
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

    MatMult(hamiltonian_basis->getMat_A(i), u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(hamiltonian_basis->getMat_A(i), v, aux); VecDot(aux, vbar, &vAvbar);
    VecSetValue(grad, i, alpha*(uAubar + vAvbar), ADD_VALUES);
  }
  int skip = hamiltonian_basis->getNBasis_A();
  for (int i=0; i<hamiltonian_basis->getNBasis_B(); i++){
    MatMult(hamiltonian_basis->getMat_B(i), u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(hamiltonian_basis->getMat_B(i), v, aux); VecDot(aux, ubar, &vBubar);
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

    /* Parameter file format: First all that corresponds to BasisMats_A = Real(-i*sigma), hence all those that correspond to purely imaginary basis matrices (for which -i*sigma) is real. Then all BasisMats_B = Imag(-i*sigma) where first come all offdiagonal ones, then all diagonal ones. */

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




HamiltonianBasis::HamiltonianBasis(int dim_rho_, bool vectorize_){
  dim_rho = dim_rho_;
  dim = dim_rho;
  vectorize = vectorize_;
  if (vectorize){
    dim = dim_rho*dim_rho; 
  }

  /* 1) Imaginary offdiagonal Gellman matrices:  sigma_jk^im = -i|j><k| + i|k><j| 
        Note: (-i)sigma_jk^IM is real, hence stored into Gellman_A = Re(H) */
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      Mat myG;
      MatCreate(PETSC_COMM_WORLD, &myG);
      MatSetType(myG, MATMPIAIJ);
      MatSetSizes(myG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      // MatMPIAIJSetPreallocation(myG, 2, NULL, 2, NULL);  // How many to allocate? Split into diag and off diag per proc.
      MatSetUp(myG);

      if (!vectorize) { // Schroedinger solver
        int row = j;
        int col = k;
        MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
        MatSetValue(myG, col, row, +1.0, INSERT_VALUES);
      } else {
        // For Lindblad: I_N \kron (-i)sigma_jk^Im - (-i)sigma_jk^Im^T \kron I_N  */
        //  -I\kron sigma_jk^Im
        for (int i=0; i<dim_rho; i++){
          int row = i*dim_rho + j;
          int col = i*dim_rho + k;
          MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, +1.0, INSERT_VALUES);
        }
        // +sigma_jk^Im^T \kron I
        for (int i=0; i<dim_rho; i++){
          int row = k*dim_rho + i;
          int col = j*dim_rho + i;
          MatSetValue(myG, row, col, 1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
        }
      }

      MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
      BasisMats_A.push_back(myG);
    }
  }

  /* 2) Real offdiagonal Gellman matrices:  sigma_jk^re = |j><k| + |k><j| 
        Note: (-i)sigma_jk^RE is purely imaginary, hence into Gellman_B = Im(H) */
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      Mat myG;
      MatCreate(PETSC_COMM_SELF, &myG);
      MatSetType(myG, MATMPIAIJ);
      MatSetSizes(myG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      // MatMPIAIJSetPreallocation(myG, 2, NULL, 2, NULL);  // How many to allocate? Split into diag and off diag per proc.
      MatSetUp(myG);

      if (!vectorize) { // Schroedinger solver
        int row = j;
        int col = k;
        MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
        MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
      } else {
        // For Lindblad: I_N \kron (-i)sigma_jk^Re - (-i)sigma_jk^Re \kron I_N  */
        //  -I\kron sigma_jk^Re
        for (int i=0; i<dim_rho; i++){
          int row = i*dim_rho + j;
          int col = i*dim_rho + k;
          MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
        }
        // +sigma_jk^Re \kron I
        for (int i=0; i<dim_rho; i++){
          int row = j*dim_rho + i;
          int col = k*dim_rho + i;
          MatSetValue(myG, row, col, 1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, 1.0, INSERT_VALUES);
        }
      }

      MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
      BasisMats_B.push_back(myG);
    }
  }

  /* 3) Real diagonal Gellman matrices:  sigma_l^Re = (2/(l(l+1))(sum_j|j><j| - l|l><l|) 
        Note: (-i)sigma_l^RE is purely imaginary, hence into Gellman_B = Im(H) */
  for (int l=1; l<dim_rho; l++){
    Mat myG;
    MatCreate(PETSC_COMM_SELF, &myG);
    MatSetType(myG, MATMPIAIJ);
    MatSetSizes(myG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    // MatMPIAIJSetPreallocation(myG, 2, NULL, 2, NULL);  // How many to allocate? Split into diag and off diag per proc.
    MatSetUp(myG);

    double prefactor = sqrt(2.0/(l*(l+1)));

    if (!vectorize) { // Schroedinger solver
      int row = l;
      MatSetValue(myG, row, row, +1.0*l*prefactor, ADD_VALUES);
      for (int j=l; j<dim_rho; j++){
        int row = j;
        MatSetValue(myG, row, row, +1.0*prefactor, ADD_VALUES);
      }
    } else { // Lindblad solver
      // first part: -I\kron sigma_l
      for (int i=0; i<dim_rho; i++){
        int row = i*dim_rho + l;
        MatSetValue(myG, row, row, +1.0*l*prefactor, ADD_VALUES);
        for (int j=l; j<dim_rho; j++){
          int row = i*dim_rho + j;
          MatSetValue(myG, row, row, +1.0*prefactor, ADD_VALUES);
        }
      }
      // second part: +sigma_l \kron I
      for (int i=0; i<dim_rho; i++){
        int row = l*dim_rho + i;
        MatSetValue(myG, row, row, -1.0*l*prefactor, ADD_VALUES);
        for (int j=l; j<dim_rho; j++){
          int row = j*dim_rho + i;
          MatSetValue(myG, row, row, -1.0*prefactor, ADD_VALUES);
        }
      }
    }

    MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
    BasisMats_B.push_back(myG);
  }

  // TEST Gellmann mats
  std::vector<Mat> G_re, G_im;
  std::vector<Mat> Test_A, Test_B;
  setupGellmannMats(dim_rho, false, G_re, G_im);
  // Now set up -isigma or vectorized -i(I kron sigma - sigma kron I)


  // setup identity matrix, if vectorizing 
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

  // First: -i*(Real_Gellmann), they go into Bd = Im(-iH)
  for (int i=0; i<G_re.size(); i++){
    Mat myMat;
    if (!vectorize){
      MatDuplicate(G_re[i],  MAT_COPY_VALUES, &myMat);
      MatScale(myMat, -1.0);
    } else {
      Mat myMat1;
      MatSeqAIJKron(G_re[i], Id, MAT_INITIAL_MATRIX, &myMat);  // sigma^T kron I
      MatSeqAIJKron(Id, G_re[i], MAT_INITIAL_MATRIX, &myMat1); // I kron sigma
      MatAXPY(myMat, -1.0, myMat1, DIFFERENT_NONZERO_PATTERN);
    }
    Test_B.push_back(myMat);
  }
  // Then: -i*(Imag_Gellmann), they go into Ad = Re(-iH) [note: no scaling by -1!]
  Mat myMat;
  for (int i=0; i<G_im.size(); i++){
    if (!vectorize){
      MatDuplicate(G_im[i],  MAT_COPY_VALUES, &myMat);
    } else {
      Mat myMat1;
      MatSeqAIJKron(G_im[i], Id, MAT_INITIAL_MATRIX, &myMat);  // sigma^T kron I
      MatSeqAIJKron(Id, G_im[i], MAT_INITIAL_MATRIX, &myMat1); // I kron sigma
      MatAXPY(myMat, 1.0, myMat1, DIFFERENT_NONZERO_PATTERN);
    }
    Test_A.push_back(myMat);
  }

  printf("Learnable basis matrices for dim=%d, dim_rho=%d, A-Mats:%d, B-Mats: %d\n", dim, dim_rho, BasisMats_A.size(), BasisMats_B.size() );
  for (int i=0; i<BasisMats_A.size(); i++){
    PetscBool equal = PETSC_FALSE;
    MatEqual(BasisMats_A[i], Test_A[i], &equal);
    printf("Basis A, equal = %d\n", equal);
    if (!equal){
      MatView(BasisMats_A[i], NULL);
      MatView(Test_A[i], NULL);
      // exit(1);
    }
  }
  for (int i=0; i<BasisMats_B.size(); i++){
    PetscBool equal = PETSC_FALSE;
    MatEqual(BasisMats_B[i], Test_B[i], &equal);
    printf("Basis B, equal = %d\n", equal);
    assert(equal);
  }
  printf("SUCCESS\n");
  // for (int i=0; i<BasisMats_A.size(); i++){
  //   printf("ORIGINAL Gellman A: i=%d\n", i);
  //   MatView(BasisMats_A[i], NULL);
  // }
  // for (int i=0; i<Test_A.size(); i++){
  //   printf("NEW TEST Gellman A: i=%d\n", i);
  //   MatView(Test_A[i], NULL);
  // }
  // for (int i=0; i<BasisMats_B.size(); i++){
  //   printf("ORIGINAL Gellman B: i=%d\n", i);
  //   MatView(BasisMats_B[i], NULL);
  // }
  // for (int i=0; i<Test_B.size(); i++){
  //   printf("NEW TEST Gellman B: i=%d\n", i);
  //   MatView(Test_B[i], NULL);
  // }
  exit(1);

  nbasis = BasisMats_A.size() + BasisMats_B.size();
}


HamiltonianBasis::~HamiltonianBasis(){

  for (int i=0; i< BasisMats_A.size(); i++){
    MatDestroy(&BasisMats_A[i]);
  }
  for (int i=0; i<BasisMats_B.size(); i++){
    MatDestroy(&BasisMats_B[i]);
  }
  BasisMats_A.clear();
  BasisMats_B.clear();
}


/* Generalized Gellmann matrices, diagonally shifted such tha G_00 = 0 */
void setupGellmannMats(int dim_rho, bool upperdiag_only, std::vector<Mat>& Gellmann_Real, std::vector<Mat>& Gellmann_Imag){

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
      if (!upperdiag_only) MatSetValue(G_re, k, j, 1.0, INSERT_VALUES);

      /* Imaginary sigma_jk^im = -i|j><k| + i|k><j| */ 
      MatSetValue(G_im, j, k, -1.0, INSERT_VALUES);
      if (!upperdiag_only) MatSetValue(G_im, k, j, +1.0, INSERT_VALUES);

      /* Assemble and store */
      MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyBegin(G_im, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(G_im, MAT_FINAL_ASSEMBLY);
      Gellmann_Real.push_back(G_re);
      Gellmann_Imag.push_back(G_im);
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
    Gellmann_Real.push_back(G_re);
  }

  // //TEST
  // printf("All REAL Gellmann matrices:\n");
  // for (int i=0; i<Gellmann_Real.size(); i++){
  //   MatView(Gellmann_Real[i], NULL);
  // }
  // printf("All IMAG Gellmann matrices:\n");
  // for (int i=0; i<Gellmann_Imag.size(); i++){
  //   MatView(Gellmann_Imag[i], NULL);
  // }
  // exit(1);
}