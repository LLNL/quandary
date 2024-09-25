#include "learning.hpp"

Learning::Learning(std::vector<int> nlevels, LindbladType lindbladtype_, UDEmodelType UDEmodel_, std::vector<std::string>& learninit_str, Data* data_, std::default_random_engine rand_engine, bool quietmode_){
  lindbladtype = lindbladtype_;
  quietmode = quietmode_;
  data = data_;
  UDEmodel = UDEmodel_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

   // Reset
  loss_integral = 0.0;
  current_err = 0.0;

  /* Get Hilbertspace dimension (dim_rho, N) and state variable (either N or N^2)*/
  dim_rho = 0;
  if (nlevels.size() > 0){
    dim_rho = 1;
    for (int i = 0; i<nlevels.size(); i++){
      dim_rho *= nlevels[i];
    }
  }
  dim = dim_rho;
  if (lindbladtype != LindbladType::NONE){
    dim = dim_rho*dim_rho; 
  }

  /* Proceed only if this is not a dummy class (aka only if using the UDE model)*/
  if (dim_rho > 0) {

    /* Create Basis for the learnable terms. */
    bool shifted_diag = false;
    if (UDEmodel == UDEmodelType::HAMILTONIAN || UDEmodel == UDEmodelType::BOTH) {
      hamiltonian_basis = new HamiltonianBasis(dim_rho, shifted_diag, lindbladtype);
    } else {
      hamiltonian_basis = new HamiltonianBasis(0, false, lindbladtype); // will be empty. Dummy
    }
    if (lindbladtype != LindbladType::NONE && (UDEmodel == UDEmodelType::LINDBLAD || UDEmodel == UDEmodelType::BOTH)) {
        lindblad_basis    = new LindbladBasis(dim_rho, shifted_diag); 
    } else {
        lindblad_basis    = new LindbladBasis(0, false);  // will be empty. Dummy
    }
    // TEST
    // lindblad_basis->showBasisMats();

    /* Set the total number of learnable paramters */
    nparams = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis();

    /* Allocate learnable Hamiltonian and Lindblad parameters, and set an initial guess */
    initLearnParams(learninit_str, rand_engine);

    /* Create auxiliary vectors needed for MatMult. */
    VecCreate(PETSC_COMM_WORLD, &aux2);    // aux2 sized for state (re and im)
    VecSetSizes(aux2, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(aux2);

    // Some output 
    if (mpirank_world == 0 && !quietmode) {
      printf("Learning with %d Gellmann mats\n", hamiltonian_basis->getNBasis());
    }
  }
}

Learning::~Learning(){
  if (dim_rho > 0) {
    learnparamsH_Re.clear();
    learnparamsH_Im.clear();
    VecDestroy(&aux2);

    delete hamiltonian_basis;
    delete lindblad_basis;
    delete data;
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


void Learning::viewOperators(std::string datadir){

  if (dim_rho <= 0) return;

  if (mpirank_world == 0) {
    bool shift_diag = true;
    if (UDEmodel == UDEmodelType::HAMILTONIAN || UDEmodel == UDEmodelType::BOTH) {

      /* Assemble the Hamiltonian, MHz, H = \sum l_i*sigma_i */
      Mat Operator_Re;  
      Mat Operator_Im;  
      MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Re);
      MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho, NULL, &Operator_Im);
      MatSetUp(Operator_Re);
      MatSetUp(Operator_Im);
      MatZeroEntries(Operator_Re);
      MatZeroEntries(Operator_Im);

      for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++) {
        MatAXPY(Operator_Re, learnparamsH_Re[i] / (2.0*M_PI), hamiltonian_basis->getBasisMat_Re(i), DIFFERENT_NONZERO_PATTERN);
      }
      for (int i=0; i<hamiltonian_basis->getNBasis_Im(); i++) {
        MatAXPY(Operator_Im, learnparamsH_Im[i] / (2.0*M_PI), hamiltonian_basis->getBasisMat_Im(i), DIFFERENT_NONZERO_PATTERN);
      }
      // If diagonally shifted: H -= H_00*Id */
      if (shift_diag) {
        double h00=0.0;
        MatGetValue(Operator_Re, 0, 0, &h00);
        MatShift(Operator_Re, -h00);
      }

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
      printf("\nLearned Lindblad system matrix written to file %s, %s\n", filename_re, filename_im);

      MatDestroy(&Operator_Re);
      MatDestroy(&Operator_Im);
    }

    if (lindbladtype != LindbladType::NONE && (UDEmodel == UDEmodelType::LINDBLAD || UDEmodel == UDEmodelType::BOTH)) {

      // print coefficients to screen
      for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
        printf("Lindblad coeff %d: %1.8e\n", i, learnparamsL_Re[i]);
      }
      if (dim_rho == 2) {
        printf(" -> maps to T_1 time %1.2f [us]\n", 1.0/learnparamsL_Re[0]);
        printf(" -> maps to T_2 time %1.2f [us]\n", 1.0/(4.0*learnparamsL_Re[1]));
      }

      /* assemble and print system matrix */
      Mat Operator;  
      MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim, dim, NULL, &Operator);
      MatSetUp(Operator);
      MatZeroEntries(Operator);
      for (int i=0; i<lindblad_basis->getSystemMats_A().size(); i++) {
        MatAXPY(Operator, learnparamsL_Re[i], lindblad_basis->getSystemMats_A()[i], DIFFERENT_NONZERO_PATTERN);
      }
      assert(lindblad_basis->getSystemMats_B().size() == 0);

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


void Learning::initLearnParams(std::vector<std::string> learninit_str, std::default_random_engine rand_engine){
  // Switch over initialization string ("file", "constant", or "random")

  if (learninit_str[0].compare("file") == 0 ) { //  Read parameter from file. 

    /* Parameter file format:  One column containing all learnable paramters as 
     *    x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
     */
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
    for (int i=0; i<hamiltonian_basis->getNBasis_Im(); i++){
      learnparamsH_Im.push_back(initguess_fromfile[i + skip]); 
    }
    // Then set all Lindblad params
    skip = hamiltonian_basis->getNBasis();
    for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
      learnparamsL_Re.push_back(initguess_fromfile[skip + i]); 
    }
    skip = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis_Re();
    for (int i=0; i<lindblad_basis->getNBasis_Im(); i++){
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
      for (int i=0; i<lindblad_basis->getNBasis_Im(); i++){
        learnparamsL_Im.push_back(amp); // ns? 
      }
    }
  } else {
    if (mpirank_world==0) printf("ERROR: Wrong configuration for learnable parameter initialization. Choose 'file, <pathtofile>', or 'random, <amplitude_Ham>, <amplitude_Lindblad>', or 'constant, <amplitude_Ham>, <amplitude_Lind>'\n");
    exit(1);
  }
}


void Learning::addToLoss(double time, Vec x, int pulse_num){

  current_err = 0.0;
  if (dim_rho <= 0) return;

  // If data point exists at this time, compute frobenius norm (x-xdata)
  Vec xdata = data->getData(time, pulse_num);
  if (xdata != NULL) {
    // printf("Add to loss at time %1.8f \n", time);
    // VecView(xdata,NULL);
    VecAYPX(aux2, 0.0, x);
    VecAXPY(aux2, -1.0, xdata);   // aux2 = x - data
    double norm; 
    VecNorm(aux2, NORM_2, &norm);
    current_err = norm;
    loss_integral += 0.5*norm*norm / (data->getNData()-1);
  }
}


void Learning::addToLoss_diff(double time, Vec xbar, Vec xprimal, int pulse_num, double Jbar_loss){

  if (dim_rho <= 0) return;

  Vec xdata = data->getData(time, pulse_num);
  if (xdata != NULL) {
    // printf("loss_DIFF at time %1.8f \n", time);
    // VecView(xprimal,NULL);
    VecAXPY(xbar, Jbar_loss / (data->getNData()-1), xprimal);
    VecAXPY(xbar, -Jbar_loss/ (data->getNData()-1), xdata);
  }
}


