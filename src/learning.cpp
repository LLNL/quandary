#include "learning.hpp"

Learning::Learning(std::vector<int>& nlevels, LindbladType lindbladtype_, std::vector<std::string>& UDEmodel_str, std::vector<int>& ncarrierwaves, std::vector<std::string>& learninit_str, Data* data_, std::default_random_engine rand_engine, bool quietmode_, double loss_scaling_factor_){
  lindbladtype = lindbladtype_;
  quietmode = quietmode_;
  data = data_;
  hamiltonian_model = NULL;
  lindblad_model = NULL;
  loss_scaling_factor=loss_scaling_factor_;
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

    // Parse the UDEmodel_str for learnable term identifyiers and create parameterizations
    for (int i=0; i<UDEmodel_str.size(); i++) {
      if (UDEmodel_str[i].compare("hamiltonian") == 0 ) {
        bool shifted_diag = true;
        hamiltonian_model = new HamiltonianModel(dim_rho, shifted_diag, lindbladtype);
      }
      if (UDEmodel_str[i].compare("lindblad") == 0 ) {
        bool shifted_diag = true;
        bool upper_only = false;
        bool real_only = false;
        lindblad_model    = new LindbladModel(dim_rho, shifted_diag, upper_only, real_only); 
      }
      if (UDEmodel_str[i].compare("transferLinear") == 0 ) {
        for (int iosc=0; iosc<ncarrierwaves.size(); iosc++){
          TransferModel* transfer_iosc = new TransferModel(dim_rho, ncarrierwaves[iosc], lindbladtype);
          transfer_model.push_back(transfer_iosc);
        }
      }
    }
    // Create dummies for those that are not used. Those will be empty, doing nothing.
    if (hamiltonian_model==NULL) hamiltonian_model = new HamiltonianModel(0, false, lindbladtype);
    if (lindblad_model==NULL) lindblad_model = new LindbladModel(0, false, false, false); 
    if (transfer_model.empty()) transfer_model.push_back(new TransferModel(0, 0, lindbladtype));

    /* Compute the total number of learnable paramters */
    nparams = hamiltonian_model->getNParams() + lindblad_model->getNParams();
    for (int iosc=0; iosc<transfer_model.size(); iosc++) {
      nparams += transfer_model[iosc]->getNParams();
    }

    /* Allocate learnable parameters, and set an initial guess */
    initLearnParams(nparams, learninit_str, rand_engine);

    /* Create auxiliary vectors needed for MatMult. */
    VecCreate(PETSC_COMM_WORLD, &aux2);    // aux2 sized for state (re and im)
    VecSetSizes(aux2, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(aux2);

    // Some output 
    // if (mpirank_world == 0 && !quietmode) {
    if (mpirank_world == 0) {
      printf("Learning with %d Hamiltonian params, %d Lindblad params, %d control transfer parameters \n", getNParamsHamiltonian(), getNParamsLindblad(), getNParamsTransfer());
    }
  }
}

Learning::~Learning(){
  if (dim_rho > 0) {
    learnparamsH.clear();
    learnparamsL.clear();
    for (int iosc=0; iosc<learnparamsT.size(); iosc++){
      learnparamsT[iosc].clear();
    }
    learnparamsT.clear();
    VecDestroy(&aux2);

    delete hamiltonian_model;
    delete lindblad_model;
    delete data;
  }
}

void Learning::applyUDESystemMats(Vec u, Vec v, Vec uout, Vec vout){

  if (dim_rho <= 0) return;

  hamiltonian_model->applySystem(u, v, uout, vout, learnparamsH);
  lindblad_model->applySystem(u, v, uout, vout, learnparamsL);
}




void Learning::applyUDESystemMats_diff(Vec u, Vec v, Vec uout, Vec vout){

  if (dim_rho <= 0) return;

  hamiltonian_model->applySystem_diff(u,v,uout, vout, learnparamsH);
  lindblad_model->applySystem_diff(u,v,uout, vout, learnparamsL);
}


void Learning::writeOperators(std::string datadir){

  if (dim_rho <= 0) return;

  if (mpirank_world == 0) {
    bool shift_diag = true;
    hamiltonian_model->writeOperator(learnparamsH, datadir);
    lindblad_model->writeOperator(learnparamsL, datadir);
  }
}


void Learning::setLearnParams(const Vec x){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, 
   *        learnparamL_Re, learnparamL_Im, 
   *        learnparamT_iosc 1,... learnparamT_iosc N] 
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
  skip += lindblad_model->getNParams();
  for (int iosc=0; iosc<transfer_model.size(); iosc++) {
    for (int i=0; i<transfer_model[iosc]->getNParams(); i++) {
      learnparamsT[iosc][i] = ptr[skip];
      skip++;
    }
  }

  VecRestoreArrayRead(x, &ptr);
}

void Learning::getLearnParams(double* x){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, 
   *        learnparamL_Re, learnparamL_Im, 
   *        learnparamT_iosc 1,... learnparamT_iosc N] 
   */

  for (int i=0; i<hamiltonian_model->getNParams(); i++) {
    x[i]      = learnparamsH[i];
  }
  int skip = hamiltonian_model->getNParams();
  for (int i=0; i<lindblad_model->getNParams(); i++) {
    x[i+skip] = learnparamsL[i];
  }
  skip += lindblad_model->getNParams();
  for (int iosc=0; iosc<transfer_model.size(); iosc++) {
    for (int i=0; i<transfer_model[iosc]->getNParams(); i++) {
      x[skip] = learnparamsT[iosc][i];
      skip++;
    }
  }
}

void Learning::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar){
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, 
   *        learnparamL_Re, learnparamL_Im, 
   *        learnparamT_iosc 1,... learnparamT_iosc N] 
   */

  if (dim_rho <= 0) return;

  hamiltonian_model->dRHSdp(grad, u, v, alpha, ubar, vbar, learnparamsH, 0);
  lindblad_model->dRHSdp(grad, u, v, alpha, ubar, vbar, learnparamsL, hamiltonian_model->getNParams());

  VecAssemblyBegin(grad);
  VecAssemblyEnd(grad);
}


void Learning::initLearnParams(int nparams, std::vector<std::string> learninit_str, std::default_random_engine rand_engine){
  // Switch over initialization string ("file", "constant", or "random")

  if (learninit_str[0].compare("file") == 0 ) { //  Read parameter from file. 

    /* Parameter file format:  One column containing all learnable paramters as 
     *    x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im, learnparamTransfer] 
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
    // Then set all transfer params, iterating over oscillators first, then over carrier waves
    skip += lindblad_model->getNParams();
    for (int iosc=0; iosc<transfer_model.size(); iosc++) {
      std::vector<double> myparams;
      for (int i=0; i<transfer_model[iosc]->getNParams(); i++){
        myparams.push_back(initguess_fromfile[skip]); 
        skip++;
      }
      learnparamsT.push_back(myparams);
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
      if (learninit_str.size() < 3) copyLast(learninit_str, 4);
      amp = atof(learninit_str[2].c_str());
      std::uniform_real_distribution<double> unit_dist2(0.0, amp);
      for (int i=0; i<lindblad_model->getNParams(); i++){
        learnparamsL.push_back(unit_dist2(rand_engine)); // 1/ns?
      }
    }
    // Then all transfer parameters. ALWAYS CONSTANT init for now.
    for (int iosc=0; iosc<transfer_model.size(); iosc++){
      if (learninit_str.size() < 4) copyLast(learninit_str, 4);
      std::vector<double> myparams;
      if (transfer_model[iosc]->getNParams() > 0) {
        amp = atof(learninit_str[3].c_str());
        for (int i=0; i<transfer_model[iosc]->getNParams(); i++){
          myparams.push_back(amp); 
        }
      }
      learnparamsT.push_back(myparams);
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
      if (learninit_str.size() < 3) copyLast(learninit_str, 4);
      amp = atof(learninit_str[2].c_str());
      for (int i=0; i<lindblad_model->getNParams(); i++){
        learnparamsL.push_back(amp); // ns?
      }
    }
    // Then all transfer parameters. For now, always init with identity. TODO.
    for (int iosc=0; iosc<transfer_model.size(); iosc++){
      std::vector<double> myparams;
      if (transfer_model[iosc]->getNParams() > 0) {
        if (learninit_str.size() < 4) copyLast(learninit_str, 4);
        amp = atof(learninit_str[3].c_str());
        for (int i=0; i<transfer_model[iosc]->getNParams(); i++){
          myparams.push_back(amp); 
        }
      }
      learnparamsT.push_back(myparams);
    }
  } else {
    if (mpirank_world==0) printf("ERROR: Wrong configuration for learnable parameter initialization. Choose 'file, <pathtofile>', or 'random, <amplitude_Ham>, <amplitude_Lindblad>, <amplitude_transfer>', or 'constant, <amplitude_Ham>, <amplitude_Lind>, <amplitude_transfer>'\n");
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
    loss_integral += loss_scaling_factor*0.5*norm*norm / (data->getNData()-1);
  }
}


void Learning::addToLoss_diff(double time, Vec xbar, Vec xprimal, int pulse_num, double Jbar_loss){

  if (dim_rho <= 0) return;

  Vec xdata = data->getData(time, pulse_num);
  if (xdata != NULL) {
    // printf("loss_DIFF at time %1.8f \n", time);
    // VecView(xprimal,NULL);
    VecAXPY(xbar, Jbar_loss  * loss_scaling_factor / (data->getNData()-1), xprimal);
    VecAXPY(xbar, -Jbar_loss * loss_scaling_factor / (data->getNData()-1), xdata);
  }
}




void Learning::applyUDETransfer(int oscilID, int cwID, double* Blt1, double* Blt2){

  if (transfer_model.size() > oscilID){
    transfer_model[oscilID]->apply(cwID, Blt1, Blt2, learnparamsT[oscilID]);
  }
}

void Learning::applyUDETransfer_diff(int oscilID, int cwID, const double Blt1, const double Blt2, double& Blt1bar, double& Blt2bar, double* grad, double x_is_control){

  if (transfer_model.size() > oscilID){
    transfer_model[oscilID]->apply_diff(cwID, Blt1, Blt2, Blt1bar, Blt2bar, grad, learnparamsT[oscilID], x_is_control);
  }
}