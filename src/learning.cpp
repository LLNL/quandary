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
    bool shifted_diag = true;
    if (UDEmodel == UDEmodelType::HAMILTONIAN || UDEmodel == UDEmodelType::BOTH) {
      hamiltonian_model = new HamiltonianModel(dim_rho, shifted_diag, lindbladtype);
    } else {
      hamiltonian_model = new HamiltonianModel(0, false, lindbladtype); // will be empty. Dummy
    }
    if (lindbladtype != LindbladType::NONE && (UDEmodel == UDEmodelType::LINDBLAD || UDEmodel == UDEmodelType::BOTH)) {
        bool upper_only = false;
        bool real_only = true;
        lindblad_model    = new LindbladModel(dim_rho, shifted_diag, upper_only, real_only); 
    } else {
        lindblad_model    = new LindbladModel(0, false, false, false);  // will be empty. Dummy
    }
    // TEST
    // lindblad_model->showBasisMats();

    /* Set the total number of learnable paramters */
    nparams = hamiltonian_model->getNParams() + lindblad_model->getNParams();

    /* Allocate learnable Hamiltonian and Lindblad parameters, and set an initial guess */
    initLearnParams(learninit_str, rand_engine);

    /* Create auxiliary vectors needed for MatMult. */
    VecCreate(PETSC_COMM_WORLD, &aux2);    // aux2 sized for state (re and im)
    VecSetSizes(aux2, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(aux2);

    // Some output 
    // if (mpirank_world == 0 && !quietmode) {
      printf("Learning with %d Hamiltonian params and %d Lindblad params \n", hamiltonian_model->getNParams(), lindblad_model->getNParams());
    // }
  }
}

Learning::~Learning(){
  if (dim_rho > 0) {
    learnparamsH.clear();
    learnparamsL.clear();
    VecDestroy(&aux2);

    delete hamiltonian_model;
    delete lindblad_model;
    delete data;
  }
}

void Learning::applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout){

  if (dim_rho <= 0) return;

  hamiltonian_model->applySystem(u, v, uout, vout, learnparamsH);
  lindblad_model->applySystem(u, v, uout, vout, learnparamsL);
}




void Learning::applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout){

  if (dim_rho <= 0) return;

  hamiltonian_model->applySystem_diff(u,v,uout, vout, learnparamsH);
  lindblad_model->applySystem_diff(u,v,uout, vout, learnparamsL);
}


void Learning::viewOperators(std::string datadir){

  if (dim_rho <= 0) return;

  if (mpirank_world == 0) {
    bool shift_diag = true;
    if (UDEmodel == UDEmodelType::HAMILTONIAN || UDEmodel == UDEmodelType::BOTH) {
      hamiltonian_model->printOperator(learnparamsH, datadir);
    }

    if (lindbladtype != LindbladType::NONE && (UDEmodel == UDEmodelType::LINDBLAD || UDEmodel == UDEmodelType::BOTH)) {
      lindblad_model->printOperator(learnparamsL, datadir);
    }
  }
}


void Learning::setLearnParams(const Vec x){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
   */

  const PetscScalar* ptr;
  VecGetArrayRead(x, &ptr);
  
  for (int i=0; i<hamiltonian_model->getNParams(); i++) {
    learnparamsH[i] = ptr[i];
  }
  int skip = hamiltonian_model->getNParams();
  for (int i=0; i<lindblad_model->getNParams(); i++) {
    learnparamsL[i] = ptr[i+skip];
  }

  VecRestoreArrayRead(x, &ptr);
}

void Learning::getLearnParams(double* x){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
   */

  for (int i=0; i<hamiltonian_model->getNParams(); i++) {
    x[i]      = learnparamsH[i];
  }
  int skip = hamiltonian_model->getNParams();
  for (int i=0; i<lindblad_model->getNParams(); i++) {
    x[i+skip] = learnparamsL[i];
  }
}

void Learning::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im ] 
   */

  if (dim_rho <= 0) return;

  hamiltonian_model->dRHSdp(grad, u, v, alpha, ubar, vbar, learnparamsH, 0);
  lindblad_model->dRHSdp(grad, u, v, alpha, ubar, vbar, learnparamsL, hamiltonian_model->getNParams());

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
    for (int i=0; i<hamiltonian_model->getNParams(); i++){
      learnparamsH.push_back(initguess_fromfile[i]); 
    }
    // Then set all Lindblad params
    int skip = hamiltonian_model->getNParams();
    for (int i=0; i<lindblad_model->getNParams(); i++){
      learnparamsL.push_back(initguess_fromfile[skip + i]); 
    }
  } else if (learninit_str[0].compare("random") == 0 ) {
    // Set uniform random parameters in [0,amp)

    // First all Hamiltonian parameters, multiply by 2*M_PI
    assert(learninit_str.size()>1);
    double amp = atof(learninit_str[1].c_str());
    std::uniform_real_distribution<double> unit_dist(0.0, amp);
    for (int i=0; i<hamiltonian_model->getNParams(); i++){
      learnparamsH.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
    }
    // Then all Lindblad parameters
    if (lindblad_model->getNParams() > 0) {
      if (learninit_str.size() == 2) learninit_str.push_back(learninit_str[1]);
      assert(learninit_str.size()>2);
      amp = atof(learninit_str[2].c_str());
      std::uniform_real_distribution<double> unit_dist2(0.0, amp);
      for (int i=0; i<lindblad_model->getNParams(); i++){
        learnparamsL.push_back(unit_dist2(rand_engine)); // ns?
      }
      // for (int i=0; i<lindblad_model->getNParams_B(); i++){
        // learnparamsL_Im.push_back(unit_dist2(rand_engine)); // ns? 
      // }
    }
  } else if (learninit_str[0].compare("constant") == 0 ) {
    // Set constant amp
    // First all Hamiltonian parameters
    assert(learninit_str.size()>1);
    double amp = atof(learninit_str[1].c_str());
    for (int i=0; i<hamiltonian_model->getNParams(); i++){
      learnparamsH.push_back(amp * 2.0*M_PI);
    }
    // Then all Lindblad parameters
    if (lindblad_model->getNParams() > 0) {
      if (learninit_str.size() == 2) learninit_str.push_back(learninit_str[1]);
      assert(learninit_str.size()>2);
      amp = atof(learninit_str[2].c_str());
      for (int i=0; i<lindblad_model->getNParams(); i++){
        learnparamsL.push_back(amp); // ns?
      }
      // for (int i=0; i<lindblad_model->getNParams_B(); i++){
        // learnparamsL_Im.push_back(amp); // ns? 
      // }
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


