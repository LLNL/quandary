#include "oscillator.hpp"

Oscillator::Oscillator(){
  nlevels = 0;
  Tfinal = 0;
  basisfunctions = NULL;
  ground_freq = 0.0;
}

Oscillator::Oscillator(int id, std::vector<int> nlevels_all_, int nbasis_, double ground_freq_, double selfkerr_, double rotational_freq_, double decay_time_, double dephase_time_, std::vector<double> carrier_freq_, double Tfinal_){

  nlevels = nlevels_all_[id];
  Tfinal = Tfinal_;
  ground_freq = ground_freq_*2.0*M_PI;
  selfkerr = selfkerr_*2.0*M_PI;
  detuning_freq = 2.0*M_PI*(ground_freq_ - rotational_freq_);
  decay_time = decay_time_;
  dephase_time = dephase_time_;

  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);
  int mpirank_world;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

  /* Create control basis functions */
  basisfunctions = new ControlBasis(nbasis_, Tfinal_, carrier_freq_);

  /* Initialize control parameters */
  int nparam = 2 * nbasis_ * carrier_freq_.size();
  for (int i=0; i<nparam; i++) {
    params.push_back(0.0);
  }

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
    delete basisfunctions;
  }
}

void Oscillator::setParams(const double* x){
  for (int i=0; i<params.size(); i++) {
    params[i] = x[i]; 
  }
}


int Oscillator::evalControl(const double t, double* Re_ptr, double* Im_ptr){

  // Sanity check 
  if ( t > Tfinal ){
    printf("ERROR: accessing spline outside of [0,T] at %f. Should never happen! Bug.\n", t);
    exit(1);
  }

  /* Evaluate the spline at time t */
  *Re_ptr = basisfunctions->evaluate(t, params, ground_freq, ControlType::RE);
  *Im_ptr = basisfunctions->evaluate(t, params, ground_freq, ControlType::IM);

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

  /* Evaluate derivative of spline basis at time t */
  double Rebar = 1.0;
  double Imbar = 1.0;
  basisfunctions->derivative(t, dRedp, Rebar, ControlType::RE);
  basisfunctions->derivative(t, dImdp, Imbar, ControlType::IM);

  /* TODO: Derivative of pipulse? */
  for (int ipulse=0; ipulse< pipulse.tstart.size(); ipulse++){
    if (pipulse.tstart[ipulse] <= t && t <= pipulse.tstop[ipulse]) {
      printf("ERROR: Derivative of pipulse not implemented. Sorry!\n");
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

  /* Evaluate the spline at time t */
  *f = basisfunctions->evaluate(t, params, ground_freq, ControlType::LAB);

  // Test implementation of lab frame controls. 
  // double forig = *f;
  // double p = basisfunctions->evaluate(t, params, ground_freq, ControlType::RE);
  // double q = basisfunctions->evaluate(t, params, ground_freq, ControlType::IM);
  // double arg = 2.0*M_PI*ground_freq*t;
  // double ftest = 2.0*p*cos(arg) - 2.0*q*sin(arg);
  // double err = fabs(forig-ftest);
  // if (err > 1e-13) printf("err %f\n", err);

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
  int dimmat = (int) sqrt(dim/2);

  /* Get locally owned portion of x */
  PetscInt ilow, iupp;
  VecGetOwnershipRange(x, &ilow, &iupp);

  /* Iterate over diagonal elements to add up expected energy level */
  double expected = 0.0;
  // YC: for-loop below can iterate only for ilow <= 2 * (i * dimmat + i) < iupp
  for (int i=0; i<dimmat; i++) {
    /* Get diagonal element in number operator */
    int num_diag = i % (nlevels*dim_postOsc);
    num_diag = num_diag / dim_postOsc;
    /* Get diagonal element in rho (real) */
    PetscInt idx_diag = getIndexReal(getVecID(i,i,dimmat));
    double xdiag = 0.0;
    if (ilow <= idx_diag && idx_diag < iupp) VecGetValues(x, 1, &idx_diag, &xdiag);
    expected += num_diag * xdiag;
  }
  
  /* Sum up from all Petsc processors */
  double myexp = expected;
  MPI_Allreduce(&myexp, &expected, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  return expected;
}


void Oscillator::expectedEnergy_diff(const Vec x, Vec x_bar, const double obj_bar) {
  PetscInt dim;
  VecGetSize(x, &dim);
  int dimmat = (int) sqrt(dim/2);
  double num_diag;

  /* Get locally owned portion of x */
  PetscInt ilow, iupp;
  VecGetOwnershipRange(x, &ilow, &iupp);

  /* Derivative of projective measure */
  for (int i=0; i<dimmat; i++) {
    int num_diag = i % (nlevels*dim_postOsc);
    num_diag = num_diag / dim_postOsc;
    PetscInt idx_diag = getIndexReal(getVecID(i, i, dimmat));
    double val = num_diag * obj_bar;
    if (ilow <= idx_diag && idx_diag < iupp) VecSetValues(x_bar, 1, &idx_diag, &val, ADD_VALUES);
  }
  VecAssemblyBegin(x_bar); VecAssemblyEnd(x_bar);

}


void Oscillator::population(const Vec x, std::vector<double> &pop) {

  int dimN = dim_preOsc * nlevels * dim_postOsc;

  assert (pop.size() == nlevels);

  std::vector<double> mypop(nlevels, 0.0);

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
        PetscInt diagID = getIndexReal(getVecID(rhoID, rhoID, dimN));  // Position in vectorized rho
        double val = 0.0;
        if (ilow <= diagID && diagID < iupp)  VecGetValues(x, 1, &diagID, &val);
        sum += val;
      }
    }
    mypop[i] = sum;
  } 

  /* Gather poppulation from all Petsc processors */
  MPI_Allreduce(mypop.data(), pop.data(), nlevels, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
}
