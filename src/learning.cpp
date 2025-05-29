#include "learning.hpp"



Learning::Learning(std::vector<int>& nlevels, LindbladType lindbladtype_, std::vector<std::string>& UDEmodel_str, std::vector<int>& ncarrierwaves, std::vector<std::string>& learninit_str, Data* data_, std::default_random_engine rand_engine, bool quietmode_, double loss_scaling_factor_) {
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
  if (nlevels.size() > 0) {
    dim_rho = 1;
    for (int i = 0; i<nlevels.size(); i++) {
      dim_rho *= nlevels[i];
    }
  }
  dim = dim_rho;
  if (lindbladtype != LindbladType::NONE) {
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
        for (int iosc=0; iosc<ncarrierwaves.size(); iosc++) {
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
} // Learning::Learning()



Learning::~Learning() {
  if (dim_rho > 0) {
    learnparamsH.clear();
    learnparamsL.clear();
    for (int iosc=0; iosc<learnparamsT.size(); iosc++) {
      for (int i=0; i<learnparamsT[iosc].size(); i++) {  
        learnparamsT[iosc][i].clear();
      }
      learnparamsT[iosc].clear();
    }
    learnparamsT.clear();
    VecDestroy(&aux2);

    delete hamiltonian_model;
    delete lindblad_model;
    delete data;
  }
} // Learning::~Learning()



int Learning::getNParamsTransfer() {
    int sum=0; 

    // Iterate over all oscillators, and sum up the number of parameters for each oscillator
    for(int iosc=0; iosc<learnparamsT.size();iosc++) { 
        sum += getNParamsTransfer(iosc); 
    } 
    
    return sum; 
} // int Learning::getNParamsTransfer() { 



int Learning::getNParamsTransfer(int iosc) {
    int sum=0; 

    // Iterate over all carrier waves, and sum up the number of parameters for each carrier wave
    for(int ncarrier = 0; ncarrier < learnparamsT[iosc].size(); ncarrier++) { 
        sum += learnparamsT[iosc][ncarrier].size(); 
    } 

    return sum; 
} // int Learning::getNParamsTransfer(int iosc)



void Learning::applyUDESystemMats(Vec u, Vec v, Vec uout, Vec vout) {

  if (dim_rho <= 0) return;

  hamiltonian_model->applySystem(u, v, uout, vout, learnparamsH);
  lindblad_model->applySystem(u, v, uout, vout, learnparamsL);
} // void Learning::applyUDESystemMats()



void Learning::applyUDESystemMats_diff(Vec u, Vec v, Vec uout, Vec vout) {

  if (dim_rho <= 0) return;

  hamiltonian_model->applySystem_diff(u,v,uout, vout, learnparamsH);
  lindblad_model->applySystem_diff(u,v,uout, vout, learnparamsL);
} // void Learning::applyUDESystemMats_diff()



void Learning::writeOperators(std::string datadir) {

  if (dim_rho <= 0) return;

  if (mpirank_world == 0) {
    bool shift_diag = true;
    hamiltonian_model->writeOperator(learnparamsH, datadir);
    lindblad_model->writeOperator(learnparamsL, datadir);
  }
} // void Learning::writeOperators()



void Learning::setLearnParams(const Vec x) {
    /* Storage of parameters in x:  
    *   x = [learnparamH_Re, learnparamH_Im, 
    *        learnparamL_Re, learnparamL_Im, 
    *        learnparamT_iosc 1,... learnparamT_iosc N] 
    * 
    * in the transfer function, for each oscillator, we first iterate over the carrier waves, and then over the parameters per carrier wave.
    */

    const PetscScalar* ptr;
    VecGetArrayRead(x, &ptr);

    // Set up "skip", which will be used to keep track of the position in the parameter vector. 
    // Because the Hamiltonian parameters are the first elements of x, skip starts at 0.
    int skip = 0;

    // Set all Hamiltonian parameters
    for (int i=0; i<hamiltonian_model->getNParams(); i++) {
        learnparamsH[i] = ptr[skip + i];
    }

    // The first element of x corresponding to the Lindblad operator comes after the last Hamiltonian parameter. 
    // Thus, we update skip to point just after the last Hamiltonian parameter.
    skip += hamiltonian_model->getNParams();

    // Set all Lindblad parameters
    for (int i=0; i<lindblad_model->getNParams(); i++) {
        learnparamsL[i] = ptr[skip + i];
    }

    // The first element of x corresponding to the transfer parameters comes after the last Lindblad parameter. 
    // Thus, we update skip to point just after the last Lindblad parameter.
    skip += lindblad_model->getNParams();

    // Finally, set all transfer parameters. In x, these are organized by oscillator, then by carrier wave, then by parameter.
    int i_transfer = 0; // Counter to keep track of where the next parameter in x is
    for (int iosc=0; iosc<transfer_model.size(); iosc++) {
        // Iterate over the carrier waves
        for (int icarrier=0; icarrier<transfer_model[iosc]->getNCarrierWaves(); icarrier++) {
            // Iterate over the parameters per carrier wave, setting them from the pointer
            for(int iparam=0; iparam<transfer_model[iosc]->getParamsPerCarrier(); iparam++) {
                learnparamsT[iosc][icarrier][iparam] = ptr[skip + i_transfer];

                // Increment the pointer (so that we are ready to read the next parameter)
                i_transfer++;
            }
        }
    }

    VecRestoreArrayRead(x, &ptr);
} // void Learning::setLearnParams()



void Learning::getLearnParams(double* x) {
  /* Storage of parameters in x:  
   *   x = [learnparamH_Re, learnparamH_Im, 
   *        learnparamL_Re, learnparamL_Im, 
   *        learnparamT_iosc 1,... learnparamT_iosc N] 
   *     
   * in the transfer function, for each oscillator, we first iterate over the carrier waves, and then over the parameters per carrier wave.
   */

    // Set up "skip", which will be used to keep track of the position in the parameter vector. 
    // Because the Hamiltonian parameters are the first elements of x, skip starts at 0.
    int skip = 0;

    // First get all Hamiltonian parameters
    for (int i=0; i<hamiltonian_model->getNParams(); i++) {
        x[skip + i]      = learnparamsH[i];
    }

    // The first element of x corresponding to the Lindblad operator comes after the last Hamiltonian parameter. 
    // Thus, we update skip to point just after the last Hamiltonian parameter.
    skip = hamiltonian_model->getNParams();
    
    // Next, get all Lindblad parameters
    for (int i=0; i<lindblad_model->getNParams(); i++) {
        x[skip + i] = learnparamsL[i];
    }

    // The first element of x corresponding to the transfer parameters comes after the last Lindblad parameter. 
    // Thus, we update skip to point just after the last Lindblad parameter.
    skip += lindblad_model->getNParams();

    // Finally, get all transfer parameters. In x, these are organized by oscillator, then by carrier wave, then by parameter.
    int i_transfer = 0; // Counter to keep track of where to put the next parameter in x
    for (int iosc=0; iosc<transfer_model.size(); iosc++) {
        // Iterate over the carrier waves
        for (int icarrier=0; icarrier<transfer_model[iosc]->getNCarrierWaves(); icarrier++) {
            // Iterate over the parameters per carrier wave, sending each to x
            for(int iparam=0; iparam<transfer_model[iosc]->getParamsPerCarrier(); iparam++) {
                x[skip + i_transfer] = learnparamsT[iosc][icarrier][iparam];

                // Increment the counter (so that we are ready to read the next parameter)
                i_transfer++;
            }
        }
    }
} // void Learning::getLearnParams()



void Learning::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar) {
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
} // void Learning::dRHSdp()



void Learning::initLearnParams(int nparams, std::vector<std::string> learninit_str, std::default_random_engine rand_engine) {
    // Switch over initialization string ("file", "constant", or "random")

    if (learninit_str[0].compare("file") == 0 ) { //  Read parameters from file. 
        // In this case, the 1 element of learninit_str is the path to the file

        /* Parameter file format:  One column containing all learnable parameters as 
        *    x = [learnparamH_Re, learnparamH_Im, learnparamL_Re, learnparamL_Im, learnparamT_iosc 1,... learnparamT_iosc N] 
        * see setLearnParams() for more details.
        */

        // Read the parameters from file
        std::vector<double> initguess_fromfile(nparams, 0.0);
        assert(learninit_str.size()>1);
        if (mpirank_world == 0) {
            read_vector(learninit_str[1].c_str(), initguess_fromfile.data(), nparams, quietmode); 
        }

        // Broadcast the parameters to all processes
        MPI_Bcast(initguess_fromfile.data(), nparams, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
        // Set up "skip", which will be used to keep track of the position in the parameter vector. 
        // Because the Hamiltonian parameters are the first elements of x, skip starts at 0.
        int skip = 0;

        // Set all Hamiltonian parameters
        for (int i=0; i<hamiltonian_model->getNParams(); i++) {
            learnparamsH.push_back(initguess_fromfile[skip + i]); 
        }

        // The first element of x corresponding to the Lindblad operator comes after the last Hamiltonian parameter. 
        // Thus, we update skip to point just after the last Hamiltonian parameter.
        skip += hamiltonian_model->getNParams();

        // Set all Lindblad parameters
        for (int i=0; i<lindblad_model->getNParams(); i++) {
            learnparamsL.push_back(initguess_fromfile[skip + i]); 
        }

        // Then set all transfer params, iterating over oscillators first, then over carrier waves
        // The first element of x corresponding to the transfer parameters comes after the last Lindblad parameter. 
        // Thus, we update skip to point just after the last Lindblad parameter.
        skip += lindblad_model->getNParams();

        // Counter to keep track of where the next parameter in x is
        int i_transfer = 0; 

        // Iterate over the oscillators
        for (int iosc=0; iosc<transfer_model.size(); iosc++) {
            // A vector to store the parameters for this oscillator; one element per carrier wave
            std::vector<std::vector<double>> myparams;

            // Iterate over the carrier waves
            for (int i=0; i<transfer_model[iosc]->getNCarrierWaves(); i++) {
                // A vector to store the parameters for this carrier wave; one element per parameter
                std::vector<double> myparams_carrier;

                // Iterate over the parameters per carrier wave, setting them from the pointer.
                for(int j = 0; j < transfer_model[iosc]->getParamsPerCarrier(); j++) {
                    myparams_carrier.push_back(initguess_fromfile[skip + i_transfer]); 
                    i_transfer++;
                }

                // Add the vector of parameters for this carrier wave to the vector of parameters for this oscillator
                myparams.push_back(myparams_carrier);
            }

            // Add the vector of parameters for this oscillator to the vector of parameters for all oscillators
            learnparamsT.push_back(myparams);
        }
    }
    
    
    
    else if (learninit_str[0].compare("random") == 0 ) { // Set uniform random parameters in [0,amp)
        /* In this case:
            learninit_str[1] is the amplitude of distribution for the Hamiltonian parameters 
            learninit_str[2] is the amplitude of distribution for the Lindblad parameters
            learninit_str[3] is the amplitude of distribution for the transfer parameters
        */

        // First get the amplitude of the distribution for the Hamiltonian parameters
        assert(learninit_str.size()>1);
        double amp = atof(learninit_str[1].c_str());

        // Set all Hamiltonian parameters, multiply the base amplitude by 2*M_PI
        std::uniform_real_distribution<double> unit_dist(0.0, amp);
        for (int i=0; i<hamiltonian_model->getNParams(); i++) {
            learnparamsH.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians?
        }

        // Set all Lindblad parameters, if there are any
        if (lindblad_model->getNParams() > 0) {
            // If the amplitude for the Lindblad parameters is not provided, use the amplitude for the Hamiltonian parameters
            if (learninit_str.size() < 3) copyLast(learninit_str, 4);
            amp = atof(learninit_str[2].c_str());

            // Set all Lindblad parameters
            std::uniform_real_distribution<double> unit_dist2(0.0, amp);
            for (int i=0; i<lindblad_model->getNParams(); i++) {
                learnparamsL.push_back(unit_dist2(rand_engine)); // 1/ns?
            }
        }

        // Set all transfer parameters. 
        // ALWAYS CONSTANT init for now. TODO: Make this configurable.
        for (int iosc=0; iosc<transfer_model.size(); iosc++) {
            // If the amplitude for the transfer parameters is not provided, use the amplitude for the Lindblad parameters
            if (learninit_str.size() < 4) copyLast(learninit_str, 4);
            amp = atof(learninit_str[3].c_str());

            // Set all transfer parameters for this oscillator, if there are any.
            // Note: This is hardcoded for a scale of 1.0 and a offset of 0. TODO: Make this configurable.
            std::vector<std::vector<double>> myparams;
            if (transfer_model[iosc]->getNParams() > 0) {
                // Iterate over the carrier waves
                for (int i=0; i<transfer_model[iosc]->getNCarrierWaves(); i++) {
                    // Set the parameters for this carrier wave
                    std::vector<double> myparams_carrier = {amp, 0.0};
                    myparams.push_back(myparams_carrier);
                }
            }

            // Add the vector of parameters for this oscillator to the vector of parameters for all oscillators
            learnparamsT.push_back(myparams);
        }
    } 
    
    
    
    else if (learninit_str[0].compare("constant") == 0 ) {
        /* In this case:
            learninit_str[1] is the amplitude of distribution for the Hamiltonian parameters 
            learninit_str[2] is the amplitude of distribution for the Lindblad parameters
            learninit_str[3] is the amplitude of distribution for the transfer parameters
        */

        // First get the amplitude of the distribution for the Hamiltonian parameters
        assert(learninit_str.size()>1);
        double amp = atof(learninit_str[1].c_str());

        // Set the Hamiltonian parameters (note that we multiply by 2*M_PI here, because the Hamiltonian parameters are in radians)
        for (int i=0; i<hamiltonian_model->getNParams(); i++) {
            learnparamsH.push_back(amp * 2.0*M_PI);
        }

        // Set the Lindblad parameters, if there are any
        if (lindblad_model->getNParams() > 0) {
            // Set up the amplitude of the distribution for the Lindblad parameters
            // If the amplitude for the Lindblad parameters is not provided, use the amplitude for the Hamiltonian parameters
            if (learninit_str.size() < 3) copyLast(learninit_str, 4);
            amp = atof(learninit_str[2].c_str());
            
            // Set the Lindblad parameters
            for (int i=0; i<lindblad_model->getNParams(); i++) {
                learnparamsL.push_back(amp); // ns?
            }
        }

        // Set the transfer parameters. 
        for (int iosc=0; iosc<transfer_model.size(); iosc++) {
            // Set up a vector to store the parameters for this oscillator (one element per carrier wave)
            std::vector<std::vector<double>> myparams;

            // If there are any transfer parameters for this oscillator, set them
            if (transfer_model[iosc]->getNParams() > 0) {
                // If the amplitude for the transfer parameters is not provided, use the amplitude for the Lindblad parameters
                if (learninit_str.size() < 4) copyLast(learninit_str, 4);
                amp = atof(learninit_str[3].c_str());

                // Iterate over the carrier waves
                for (int i=0; i<transfer_model[iosc]->getNCarrierWaves(); i++) {
                    // Set the parameters for this carrier wave
                    // Note: This is hardcoded for a scale of amp and a offset of 0. 
                    // TODO: Make this configurable.
                    std::vector<double> myparams_carrier = {amp, 0.0};
                    myparams.push_back(myparams_carrier);
                }
            }

            // Add the vector of parameters for this oscillator to the vector of parameters for all oscillators
            learnparamsT.push_back(myparams);
        }

    } 
    
    
    else {
        if (mpirank_world==0) printf("ERROR: Wrong configuration for learnable parameter initialization. Choose 'file, <pathtofile>', or 'random, <amplitude_Ham>, <amplitude_Lindblad>, <amplitude_transfer>', or 'constant, <amplitude_Ham>, <amplitude_Lind>, <amplitude_transfer>'\n");
        exit(1);
    }
} // void Learning::initLearnParams()



void Learning::addToLoss(double time, Vec x, int pulse_num) {

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
} // void Learning::addToLoss()



void Learning::addToLoss_diff(double time, Vec xbar, Vec xprimal, int pulse_num, double Jbar_loss) {

  if (dim_rho <= 0) return;

  Vec xdata = data->getData(time, pulse_num);
  if (xdata != NULL) {
    // printf("loss_DIFF at time %1.8f \n", time);
    // VecView(xprimal,NULL);
    VecAXPY(xbar, Jbar_loss  * loss_scaling_factor / (data->getNData()-1), xprimal);
    VecAXPY(xbar, -Jbar_loss * loss_scaling_factor / (data->getNData()-1), xdata);
  }
} // void Learning::addToLoss_diff()



void Learning::applyUDETransfer(int oscilID, int cwID, double* Blt1, double* Blt2) {

  if (transfer_model.size() > oscilID) {
    transfer_model[oscilID]->apply(cwID, Blt1, Blt2, learnparamsT[oscilID]);
  }
} // void Learning::applyUDETransfer()



void Learning::applyUDETransfer_diff(int oscilID, int cwID, const double Blt1, const double Blt2, double& Blt1bar, double& Blt2bar, double* grad, double x_is_control) {

  if (transfer_model.size() > oscilID) {
    transfer_model[oscilID]->apply_diff(cwID, Blt1, Blt2, Blt1bar, Blt2bar, grad, learnparamsT[oscilID], x_is_control);
  }
}