#include "learning.hpp"

Learning::Learning(int dim_rho_, LindbladType lindbladtype_, std::vector<std::string>& learninit_str, Data* data_, std::default_random_engine rand_engine, bool quietmode_){
  lindbladtype = lindbladtype_;
  quietmode = quietmode_;
  data = data_;
  dim_rho = dim_rho_;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  loss_integral = 0.0;

  // Get dimension of the state variable (dim = N or N^2 for Schroedinger or Lindblad solver)
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


void Learning::viewOperators(){

  if (dim_rho <= 0) return;

  hamiltonian_basis->assembleOperator(learnparamsH_Re, learnparamsH_Im);
  printf("\nLearned Hamiltonian operator: Re = \n");
  MatView(hamiltonian_basis->getOperator_Re(), NULL);
  printf("Learned Hamiltonian operator: Im = \n");
  MatView(hamiltonian_basis->getOperator_Im(), NULL);

  for (int i=0; i<lindblad_basis->getNBasis_Re(); i++){
    printf("Lindblad: %d\n", i);
    MatScale(lindblad_basis->getBasisMat_Re(i), sqrt(learnparamsL_Re[i]));
    MatView(lindblad_basis->getBasisMat_Re(i), NULL);
    // Revert scaling, just to be safe...
    MatScale(lindblad_basis->getBasisMat_Re(i), 1.0/sqrt(learnparamsL_Re[i]));
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
    int nparams = hamiltonian_basis->getNBasis() + lindblad_basis->getNBasis();
    std::vector<double> initguess_fromfile(nparams, 0.0);
    assert(learninit_str.size()>1);
    if (mpirank_world == 0) {
      read_vector(learninit_str[1].c_str(), initguess_fromfile.data(), nparams, quietmode);
    }
    MPI_Bcast(initguess_fromfile.data(), nparams, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
    //First set all Hamiltonian parameters
    for (int i=0; i<hamiltonian_basis->getNBasis_Re(); i++){
      //learnparamsH_Re.push_back(initguess_fromfile[i]/(2*M_PI)); 
      learnparamsH_Re.push_back(initguess_fromfile[i]); 
    }
    int skip = hamiltonian_basis->getNBasis_Re();
    for (int i=0; i<=hamiltonian_basis->getNBasis_Im(); i++){
      // learnparamsH_Im.push_back(initguess_fromfile[i + skip]/(2*M_PI)); 
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


void Learning::addToLoss(double time, Vec x){

  if (dim_rho <= 0) return;

  // If data point exists at this time, compute frobenius norm (x-xdata)
  Vec xdata = data->getData(time);
  if (xdata != NULL) {
    // printf("Add to loss at time %1.8f \n", time);
    // VecView(xdata,NULL);
    VecAYPX(aux2, 0.0, x);
    VecAXPY(aux2, -1.0, xdata);   // aux2 = x - data
    double norm; 
    VecNorm(aux2, NORM_2, &norm);
    loss_integral += 0.5*norm*norm / (data->getNData()-1);
  }
}


void Learning::addToLoss_diff(double time, Vec xbar, Vec xprimal, double Jbar_loss){

  if (dim_rho <= 0) return;

  Vec xdata = data->getData(time);
  if (xdata != NULL) {
    // printf("loss_DIFF at time %1.8f \n", time);
    // VecView(xprimal,NULL);
    VecAXPY(xbar, Jbar_loss / (data->getNData()-1), xprimal);
    VecAXPY(xbar, -Jbar_loss/ (data->getNData()-1), xdata);
  }
}


