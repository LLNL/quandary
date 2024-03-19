#include "oscillator.hpp"

Oscillator::Oscillator(){
  nlevels = 0;
  Tfinal = 0;
  ground_freq = 0.0;
  control_enforceBC = true;
}

Oscillator::Oscillator(MapParam config, int id, std::vector<int> nlevels_all_, std::vector<std::string>& controlsegments, std::vector<std::string>& controlinitializations, double ground_freq_, double selfkerr_, double rotational_freq_, double decay_time_, double dephase_time_, std::vector<double> carrier_freq_, double Tfinal_, LindbladType lindbladtype_, std::default_random_engine rand_engine){

  myid = id;
  nlevels = nlevels_all_[id];
  Tfinal = Tfinal_;
  ground_freq = ground_freq_*2.0*M_PI;
  selfkerr = selfkerr_*2.0*M_PI;
  detuning_freq = 2.0*M_PI*(ground_freq_ - rotational_freq_);
  carrier_freq = carrier_freq_;
  for (int i=0; i<carrier_freq.size(); i++) {
    carrier_freq[i] *= 2.0*M_PI;
  }
  decay_time = decay_time_;
  dephase_time = dephase_time_;
  lindbladtype = lindbladtype_;

  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);


  // for (int idstr = 0; idstr < controlsegments.size(); idstr++) printf("%s ", controlsegments[idstr].c_str());
  // printf("\n");

  // Parse for control segments
  int idstr = 0;
  int nparams = 0;
  while (idstr < controlsegments.size()) {

    if (controlsegments[idstr].compare("step") == 0) {
      idstr++;
      if (controlsegments.size() <= idstr+2){
        printf("ERROR: Wrong setting for control segments: Step Amplitudes or tramp not found.\n");
        exit(1);
      }
      double step_amp1 = atof(controlsegments[idstr].c_str()); idstr++;
      double step_amp2 = atof(controlsegments[idstr].c_str()); idstr++;
      double tramp = atof(controlsegments[idstr].c_str()); idstr++;

      double tstart = 0.0;
      double tstop = Tfinal;
      if (controlsegments.size()>=idstr+2){
        tstart = atof(controlsegments[idstr].c_str()); idstr++;
        tstop = atof(controlsegments[idstr].c_str()); idstr++;
      }
      // if (mpirank_world == 0) printf("%d: Creating step basis with amplitude (%f, %f) (tramp %f) in control segment [%f, %f]\n", myid, step_amp1, step_amp2, tramp, tstart, tstop);
      ControlBasis* mystep = new Step(step_amp1, step_amp2, tstart, tstop, tramp);
      mystep->setSkip(nparams);
      nparams += mystep->getNparams() * carrier_freq.size();
      basisfunctions.push_back(mystep);
      
    } else if (controlsegments[idstr].compare("spline") == 0) { // Default: splines. Format in string: spline, nsplines, tstart, tstop
      idstr++;
      if (controlsegments.size() <= idstr){
        printf("ERROR: Wrong setting for control segments: Number of splines not found.\n");
        exit(1);
      }
      int nspline = atoi(controlsegments[idstr].c_str()); idstr++;
      double tstart = 0.0;
      double tstop = Tfinal;
      if (controlsegments.size()>=idstr+2){
        tstart = atof(controlsegments[idstr].c_str()); idstr++;
        tstop = atof(controlsegments[idstr].c_str()); idstr++;
      }
      // if (mpirank_world==0) printf("%d: Creating %d-spline basis in control segment [%f, %f]\n", myid, nspline,tstart, tstop);
      ControlBasis* mysplinebasis = new BSpline2nd(nspline, tstart, tstop);
      mysplinebasis->setSkip(nparams);
      nparams += mysplinebasis->getNparams() * carrier_freq.size();
      basisfunctions.push_back(mysplinebasis);
    } else if (controlsegments[idstr].compare("spline_amplitude") == 0) { // Spline on amplitude only. Format in string: spline_amplitude, nsplines, tstart, tstop
      idstr++;
      if (controlsegments.size() <= idstr){
        printf("ERROR: Wrong setting for control segments: Number of splines not found.\n");
        exit(1);
      }
      int nspline = atoi(controlsegments[idstr].c_str()); idstr++;
      double scaling = atof(controlsegments[idstr].c_str()); idstr++;
      double tstart = 0.0;
      double tstop = Tfinal;
      if (controlsegments.size()>=idstr+2){
        tstart = atof(controlsegments[idstr].c_str()); idstr++;
        tstop = atof(controlsegments[idstr].c_str()); idstr++;
      }
      // if (mpirank_world==0) printf("%d: Creating %d-spline basis in control segment [%f, %f]\n", myid, nspline,tstart, tstop);
      ControlBasis* mysplinebasis = new BSpline2ndAmplitude(nspline, scaling, tstart, tstop);
      mysplinebasis->setSkip(nparams);
      nparams += mysplinebasis->getNparams() * carrier_freq.size();
      basisfunctions.push_back(mysplinebasis);
    } else {
      // if (mpirank_world==0) printf("%d: Non-controllable.\n", myid);
      idstr++;
    }
  }

  //Initialization of the control parameters for each segment
  int idini = 0;
  for (int seg = 0; seg < basisfunctions.size(); seg++) {
    // Set a default if initialization string is not given for this segment
    if (controlinitializations.size() < idini+2) {
      controlinitializations.push_back("constant");
      if (basisfunctions[seg]->getType() == ControlType::STEP)
        controlinitializations.push_back("1.0");
      else 
        controlinitializations.push_back("0.0");
    }
    // Check config option for 'constant' or 'random' initialization
    double initval = atof(controlinitializations[idini+1].c_str())*2.0*M_PI;
    if (controlinitializations[idini].compare("constant") == 0 ) {
      // If STEP: scale to [0,1]
      if (basisfunctions[seg]->getType() == ControlType::STEP){
        initval = std::max(0.0, initval);  
        initval = std::min(1.0, initval); 
      }
      for (int f = 0; f<carrier_freq.size(); f++) {
        for (int i=0; i<basisfunctions[seg]->getNparams(); i++){
          params.push_back(initval);
        }
        // if BSPLINEAMP: Two values can be provided: First one for the amplitude (set above), second one for the phase which otherwise is set to 0.0 (overwrite here)
        if (basisfunctions[seg]->getType() == ControlType::BSPLINEAMP) {
          if (controlinitializations.size() > idini+2) params[params.size()-1] = atof(controlinitializations[idini+2].c_str());
          else params[params.size()-1] = 0.0;
        }
      }
    } else if (controlinitializations[idini].compare("random") == 0) {

      // Uniform distribution [0,1)
      std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

      for (int f = 0; f<carrier_freq.size(); f++) {
        for (int i=0; i<basisfunctions[seg]->getNparams(); i++){
          double randval = unit_dist(rand_engine);  // random in [0,1]
          // scale to chosen amplitude 
          double val = initval*randval;

          // If STEP: scale to [0,1] else scale to [-a,a]
          if (basisfunctions[seg]->getType() == ControlType::STEP){
            val = std::max(0.0, val);  
            val = std::min(1.0, val); 
          } else {
            val = 2*val - initval;
          }
          params.push_back(val);
        }
        // if BSPLINEAMP: Two values can be provided: First one for the amplitude (set above), second one for the phase which otherwise is set to 0.0 (overwrite here)
        if (basisfunctions[seg]->getType() == ControlType::BSPLINEAMP) {
          if (controlinitializations.size() > idini+2) params[params.size()-1] = atof(controlinitializations[idini+2].c_str());
          else params[params.size()-1] = 0.0;
        }
      }
    } else {
      for (int i=0; i<basisfunctions[seg]->getNparams() * carrier_freq.size(); i++){
        params.push_back(0.0);
      }
    }
    idini += 2; 
  }


  /* Check if boundary conditions for controls should be enfored (default: yes). */
  control_enforceBC = config.GetBoolParam("control_enforceBC", true);

  /* Compute and store dimension of preceding and following oscillators */
  dim_preOsc = 1;
  dim_postOsc = 1;
  for (int j=0; j<nlevels_all_.size(); j++) {
    if (j < id) dim_preOsc  *= nlevels_all_[j];
    if (j > id) dim_postOsc *= nlevels_all_[j];
  }

}


Oscillator::~Oscillator(){
  if (params.size() > 0) {
    for (int i=0; i<basisfunctions.size(); i++) 
      delete basisfunctions[i];
  }
}

void Oscillator::setParams(double* x){

  if (params.size() > 0 && control_enforceBC){
    //printf("\n\n True! \n\n");
    // First, enforce the control boundaries, i.e. potentially set some parameters in x to zero. 
    for (int bs = 0; bs < basisfunctions.size(); bs++){
      for (int f=0; f < carrier_freq.size(); f++) {
        basisfunctions[bs]->enforceBoundary(x, f);
      }
    }
  }

  // Now copy x into the oscillators parameter storage
  for (int i=0; i<params.size(); i++) {
    params[i] = x[i]; 
  }

}


void Oscillator::setParams_diff(double* xbar){

  if (params.size() > 0 && control_enforceBC){
    //printf("\n\n True! \n\n");
    // First, enforce the control boundaries, i.e. potentially set some parameters in x to zero. 
    for (int bs = 0; bs < basisfunctions.size(); bs++){
      for (int f=0; f < carrier_freq.size(); f++) {
        basisfunctions[bs]->enforceBoundary(xbar, f);
      }
    }
  }
}


void Oscillator::getParams(double* x){
  for (int i=0; i<params.size(); i++) {
    x[i] = params[i]; 
  }
}

int Oscillator::getNSegParams(int segmentID){
  int n = 0;
  if (params.size()>0) {
    assert(basisfunctions.size() > segmentID);
    n = basisfunctions[segmentID]->getNparams()*carrier_freq.size();
  }
  return n; 
}


int Oscillator::evalControl(const double t, double* Re_ptr, double* Im_ptr){

  // Sanity check 
  if ( t > Tfinal ){
    printf("ERROR: accessing spline outside of [0,T] at %f. Should never happen! Bug.\n", t);
    exit(1);
  }

  // Default: Non controllable oscillator. Will typically be overwritten below. 
  *Re_ptr = 0.0;
  *Im_ptr = 0.0;

  /* Evaluate p(t) and q(t) using the parameters */
  if (params.size()>0) {
    // Iterate over basis parameterizations. Only one will be used, see the break-statement. 
    for (int bs = 0; bs < basisfunctions.size(); bs++){
      if (basisfunctions[bs]->getTstart() <= t && 
          basisfunctions[bs]->getTstop() >= t ) {
        /* Iterate over carrier frequencies */
        double sum_p = 0.0;
        double sum_q = 0.0;
        for (int f=0; f < carrier_freq.size(); f++) {
          double Blt1 = 0.0; 
          double Blt2 = 0.0;
          basisfunctions[bs]->evaluate(t, params, f, &Blt1, &Blt2);
          if (basisfunctions[bs]->getType() == ControlType::BSPLINEAMP) {
            double cos_omt = cos(carrier_freq[f]*t + Blt2);
            double sin_omt = sin(carrier_freq[f]*t + Blt2);
            sum_p += cos_omt * Blt1; 
            sum_q += sin_omt * Blt1;
          } else {
            double cos_omt = cos(carrier_freq[f]*t);
            double sin_omt = sin(carrier_freq[f]*t);
            sum_p += cos_omt * Blt1 - sin_omt * Blt2; 
            sum_q += sin_omt * Blt1 + cos_omt * Blt2;
          }
        }
        *Re_ptr = sum_p;
        *Im_ptr = sum_q;
        break;
      }
    }
  } 

  /* If pipulse: Overwrite controls by constant amplitude */
  for (int ipulse=0; ipulse< pipulse.tstart.size(); ipulse++){
    if (pipulse.tstart[ipulse] <= t && t <= pipulse.tstop[ipulse]) {
      double amp_pq =  pipulse.amp[ipulse] / sqrt(2.0);
      *Re_ptr = amp_pq;
      *Im_ptr = amp_pq;
    }
  }

  return 0;
}

int Oscillator::evalControl_diff(const double t, double* dRedp, double* dImdp) {

  // Sanity check 
  if ( t > Tfinal ){
    printf("ERROR: accessing spline outside of [0,T] at %f. Should never happen! Bug.\n", t);
    exit(1);
  } 


  if (params.size()>0) {
    // Iterate over basis parameterizations
    for (int bs = 0; bs < basisfunctions.size(); bs++){
      if (basisfunctions[bs]->getTstart() <= t && 
          basisfunctions[bs]->getTstop() >= t ) {
        /* Iterate over carrier frequencies */
        for (int f=0; f < carrier_freq.size(); f++) {

          if (basisfunctions[bs]->getType() == ControlType::BSPLINEAMP) {
            basisfunctions[bs]->derivative(t, params, dRedp, carrier_freq[f], 1.0, f);  // +/-1.0 is used as a flag inside Bsline2ndAmplitude->evaluate() to determine whether this is for p (1.0) or for q (-1.0)
            basisfunctions[bs]->derivative(t, params, dImdp, carrier_freq[f], -1.0, f);
          } else {
            double cos_omt = cos(carrier_freq[f]*t);
            double sin_omt = sin(carrier_freq[f]*t);
            basisfunctions[bs]->derivative(t, params, dRedp, cos_omt, -sin_omt, f);
            basisfunctions[bs]->derivative(t, params, dImdp, sin_omt, cos_omt, f);
          }
        }
        break;
      }
    }
  } 

  /* TODO: Derivative of pipulse? */
  for (int ipulse=0; ipulse< pipulse.tstart.size(); ipulse++){
    if (pipulse.tstart[ipulse] <= t && t <= pipulse.tstop[ipulse]) {
      printf("ERROR: Derivative of pipulse not implemented. Sorry! But also, this should never happen!\n");
      exit(1);
    }
  }

  return 0;
}

int Oscillator::evalControl_Labframe(const double t, double* f){

  // Sanity check 
  if ( t > Tfinal ){
    printf("ERROR: accessing spline outside of [0,T] at %f. Should never happen! Bug.\n", t);
    exit(1);
  }

  /* Evaluate the spline at time t  */
  *f = 0.0;
  if (params.size()>0) {
    // Iterate over basis parameterizations
    for (int bs = 0; bs < basisfunctions.size(); bs++){
      if (basisfunctions[bs]->getTstart() <= t && 
          basisfunctions[bs]->getTstop() >= t ) {
        /* Iterate over carrier frequencies multiply with basisfunction of index k0 */
        double sum_p = 0.0;
        double sum_q = 0.0;
        for (int f=0; f < carrier_freq.size(); f++) {
          double cos_omt = cos(carrier_freq[f]*t);
          double sin_omt = sin(carrier_freq[f]*t);
          double Blt1 = 0.0; 
          double Blt2 = 0.0;
          basisfunctions[bs]->evaluate(t, params, f, &Blt1, &Blt2);
          sum_p += cos_omt * Blt1 - sin_omt * Blt2; 
          sum_q += sin_omt * Blt1 + cos_omt * Blt2;
        }
        *f = 2. * (sum_p * cos(ground_freq*t) - sum_q * sin(ground_freq*t));
        break;
      }
    }
  } 



  /* If inside a pipulse, overwrite lab control */
  for (int ipulse=0; ipulse< pipulse.tstart.size(); ipulse++){
    if (pipulse.tstart[ipulse] <= t && t <= pipulse.tstop[ipulse]) {
      double p = pipulse.amp[ipulse] / sqrt(2.0);
      double q = pipulse.amp[ipulse] / sqrt(2.0);
      *f = 2.0 * p * cos(ground_freq*t) - 2.0 * q * sin(ground_freq*t);
    }
  }

  return 0;
}

double Oscillator::expectedEnergy(const Vec x) {
 
  PetscInt dim;
  VecGetSize(x, &dim);
  int dimmat;
  if (lindbladtype != LindbladType::NONE)  dimmat = (int) sqrt(dim/2);
  else dimmat = (int) dim/2;

  /* Get locally owned portion of x */
  PetscInt ilow, iupp, idx_diag_re, idx_diag_im;
  VecGetOwnershipRange(x, &ilow, &iupp);

  /* Iterate over diagonal elements to add up expected energy level */
  double expected = 0.0;
  // YC: for-loop below can iterate only for ilow <= 2 * (i * dimmat + i) < iupp
  for (int i=0; i<dimmat; i++) {
    /* Get diagonal element in number operator */
    int num_diag = i % (nlevels*dim_postOsc);
    num_diag = num_diag / dim_postOsc;
    /* Get diagonal element in rho (real) */
    if (lindbladtype != LindbladType::NONE) idx_diag_re = getIndexReal(getVecID(i,i,dimmat));
    else {
      idx_diag_re = getIndexReal(i);
      idx_diag_im = getIndexImag(i);
    }
    
    double xdiag = 0.0;
    if (lindbladtype != LindbladType::NONE){ // Lindblad solver: += i * rho_ii
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecGetValues(x, 1, &idx_diag_re, &xdiag);
      expected += num_diag * xdiag;
    }
    else { // Schoedinger solver: += i * | psi_i |^2
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecGetValues(x, 1, &idx_diag_re, &xdiag);
      expected += num_diag * xdiag * xdiag;
      if (ilow <= idx_diag_im && idx_diag_im < iupp) VecGetValues(x, 1, &idx_diag_im, &xdiag);
      expected += num_diag * xdiag * xdiag;
    }
  }
  
  /* Sum up from all Petsc processors */
  // double myexp = expected;
  // MPI_Allreduce(&myexp, &expected, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  return expected;
}


void Oscillator::expectedEnergy_diff(const Vec x, Vec x_bar, const double obj_bar) {
  PetscInt dim;
  VecGetSize(x, &dim);
  int dimmat;
  if (lindbladtype != LindbladType::NONE) dimmat = (int) sqrt(dim/2);
  else dimmat = (int) dim/2;
  double num_diag, xdiag, val;

  /* Get locally owned portion of x */
  PetscInt ilow, iupp, idx_diag_re, idx_diag_im;
  VecGetOwnershipRange(x, &ilow, &iupp);

  /* Derivative of projective measure */
  for (int i=0; i<dimmat; i++) {
    int num_diag = i % (nlevels*dim_postOsc);
    num_diag = num_diag / dim_postOsc;
    if (lindbladtype != LindbladType::NONE) { // Lindblas solver
      val = num_diag * obj_bar;
      idx_diag_re = getIndexReal(getVecID(i, i, dimmat));
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecSetValues(x_bar, 1, &idx_diag_re, &val, ADD_VALUES);
    }
    else {
      // Real part
      idx_diag_re = getIndexReal(i);
      xdiag = 0.0;
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecGetValues(x, 1, &idx_diag_re, &xdiag);
      val = num_diag * xdiag * obj_bar;
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecSetValues(x_bar, 1, &idx_diag_re, &val, ADD_VALUES);
      // Imaginary part
      idx_diag_im = getIndexImag(i);
      xdiag = 0.0;
      if (ilow <= idx_diag_im && idx_diag_im < iupp) VecGetValues(x, 1, &idx_diag_im, &xdiag);
      val = - num_diag * xdiag * obj_bar; // TODO: Is this a minus or a plus?? 
      if (ilow <= idx_diag_im && idx_diag_im < iupp) VecSetValues(x_bar, 1, &idx_diag_im, &val, ADD_VALUES);
    }
  }
  VecAssemblyBegin(x_bar); VecAssemblyEnd(x_bar);

}


void Oscillator::population(const Vec x, std::vector<double> &pop) {

  int dimN = dim_preOsc * nlevels * dim_postOsc;
  double val;

  assert (pop.size() == nlevels);

  // std::vector<double> mypop(nlevels, 0.0);

  /* Get locally owned portion of x */
  PetscInt ilow, iupp;
  VecGetOwnershipRange(x, &ilow, &iupp);

  /* Iterate over diagonal elements of the reduced density matrix for this oscillator */
  for (int i=0; i < nlevels; i++) {
    int identitystartID = i * dim_postOsc;
    /* Sum up elements from all dim_preOsc blocks of size (n_k * dim_postOsc) */
    double sum = 0.0;
    for (int j=0; j < dim_preOsc; j++) {
      int blockstartID = j * nlevels * dim_postOsc; // Go to the block
      /* Iterate over identity */
      for (int l=0; l < dim_postOsc; l++) {
        /* Get diagonal element */
        int rhoID = blockstartID + identitystartID + l; // Diagonal element of rho
        if (lindbladtype != LindbladType::NONE) { // Lindblad solver
          PetscInt diagID = getIndexReal(getVecID(rhoID, rhoID, dimN));  // Position in vectorized rho
          double val = 0.0;
          if (ilow <= diagID && diagID < iupp)  VecGetValues(x, 1, &diagID, &val);
          sum += val;
        } else {
          PetscInt diagID_re = getIndexReal(rhoID);
          PetscInt diagID_im = getIndexImag(rhoID);
          val = 0.0;
          if (ilow <= diagID_re && diagID_re < iupp)  VecGetValues(x, 1, &diagID_re, &val);
          sum += val * val;
          val = 0.0;
          if (ilow <= diagID_im && diagID_im < iupp)  VecGetValues(x, 1, &diagID_im, &val);
          sum += val * val;
        }
      }
    }
    pop[i] = sum;
  } 

  /* Gather poppulation from all Petsc processors */
  // for (int i=0; i<mypop.size(); i++) {pop[i] = mypop[i];}
  // MPI_Allreduce(mypop.data(), pop.data(), nlevels, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
}
