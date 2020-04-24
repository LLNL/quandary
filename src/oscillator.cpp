#include "oscillator.hpp"

Oscillator::Oscillator(){
  nlevels = 0;
  Tfinal = 0;
  basisfunctions = NULL;
}

Oscillator::Oscillator(int id, std::vector<int> nlevels_all_, int nbasis_, std::vector<double> carrier_freq_, double Tfinal_){
  nlevels = nlevels_all_[id];
  Tfinal = Tfinal_;
  basisfunctions = new ControlBasis(nbasis_, Tfinal_, carrier_freq_);
  printf("Creating oscillator with %d levels, %d carrierwave frequencies: ", nlevels, carrier_freq_.size());
  for (int f = 0; f < carrier_freq_.size(); f++) {
    printf("%f ", carrier_freq_[f]);
  }
  printf("\n");


  /* Create and store the number and lowering operators */
  dim_preOsc = 1;
  dim_postOsc = 1;
  for (int j=0; j<nlevels_all_.size(); j++) {
    if (j < id) dim_preOsc  *= nlevels_all_[j];
    if (j > id) dim_postOsc *= nlevels_all_[j];
  }
  createNumberOP(dim_preOsc, dim_postOsc, &NumberOP);
  createLoweringOP(dim_preOsc, dim_postOsc, &LoweringOP);


  /* Initialize control parameters */
  int nparam = 2 * nbasis_ * carrier_freq_.size();
  for (int i=0; i<nparam; i++) {
    params.push_back(0.0);
  }
}


Oscillator::~Oscillator(){
  if (params.size() > 0) {
    delete basisfunctions;
    MatDestroy(&NumberOP);
    MatDestroy(&LoweringOP);
  }
}

Mat Oscillator::getNumberOP() {
  return NumberOP;
}

Mat Oscillator::getLoweringOP() {
  return LoweringOP;
}

void Oscillator::flushControl(int ntime, double dt, const char* filename) {
  double time;
  double Re, Im;
  FILE *file = 0;
  file = fopen(filename, "w");

  for (int i=0; i<ntime; i++) {
    time = i*dt; 
    this->evalControl(time, &Re, &Im);
    fprintf(file, "%08d  % 1.4f   % 1.14e   % 1.14e\n", i, time, Re, Im);
  }

  fclose(file);
  printf("File written: %s\n", filename);
}

void Oscillator::setParams(const double* x){
  for (int i=0; i<params.size(); i++) {
    params[i] = x[i]; 
  }
}

void Oscillator::getParams(double* x){
  for (int i=0; i<params.size(); i++) {
    x[i] = params[i];
  }
}


int Oscillator::createNumberOP(int dim_prekron, int dim_postkron, Mat* numberOP) {

  int dim_number = dim_prekron*nlevels*dim_postkron;

  /* Create and set number operator */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim_number, dim_number,dim_number,NULL, numberOP); 
  for (int i=0; i<dim_prekron; i++) {
    for (int j=0; j<nlevels; j++) {
      double val = j;
      for (int k=0; k<dim_postkron; k++) {
        int row = i * nlevels*dim_postkron + j * dim_postkron + k;
        int col = row;
        MatSetValue(*numberOP, row, col, val, INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(*numberOP, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*numberOP, MAT_FINAL_ASSEMBLY);
 
  return dim_number;
}



int Oscillator::createLoweringOP(int dim_prekron, int dim_postkron, Mat* loweringOP) {

  int dim_lowering = dim_prekron*nlevels*dim_postkron;

  /* create and set lowering operator */
  MatCreateSeqAIJ(PETSC_COMM_WORLD,dim_lowering,dim_lowering,dim_lowering-1,NULL, loweringOP); 
  for (int i=0; i<dim_prekron; i++) {
    for (int j=0; j<nlevels-1; j++) {
      double val = sqrt(j+1);
      for (int k=0; k<dim_postkron; k++) {
        int row = i * nlevels*dim_postkron + j * dim_postkron + k;
        int col = row + dim_postkron;
        MatSetValue(*loweringOP, row, col, val, INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(*loweringOP, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*loweringOP, MAT_FINAL_ASSEMBLY);

  return dim_lowering;
}


int Oscillator::evalControl(double t, double* Re_ptr, double* Im_ptr){

  if ( t > Tfinal ){
    printf("WARNING: accessing spline outside of [0,T] at %f. Returning 0.0\n", t);
    *Re_ptr = 0.0;
    *Im_ptr = 0.0;
  } else {
    /* Evaluate the spline at time t */
    *Re_ptr = basisfunctions->evaluate(t, params, ControlBasis::RE);
    *Im_ptr = basisfunctions->evaluate(t, params, ControlBasis::IM);
  }

  return 0;
}


int Oscillator::evalDerivative(double t, double* dRedp, double* dImdp) {

  if ( t > Tfinal ){
    printf("WARNING: accessing spline derivative outside of [0,T]. Returning 0.0\n");
    for (int i = 0; i < params.size(); i++) {
      dRedp[i] = 0.0;
      dImdp[i] = 0.0;
    }
  } else {
      double Rebar = 1.0;
      double Imbar = 1.0;
      basisfunctions->derivative(t, dRedp, Rebar, ControlBasis::RE);
      basisfunctions->derivative(t, dImdp, Imbar, ControlBasis::IM);
  }

  return 0;
}


double Oscillator::expectedEnergy(Vec x) {

  int dimmat;
  MatGetSize(NumberOP, &dimmat, NULL);
  double xdiag, num_diag;

  double expected = 0.0;
  for (int i=0; i<dimmat; i++) {
    /* Get diagonal element in number operator */
    MatGetValue(NumberOP, i, i, &num_diag);
    /* Get diagonal element in rho */
    int idx_diag = i * dimmat + i;
    VecGetValues(x, 1, &idx_diag, &xdiag);
    expected += num_diag * xdiag;
  }

  return expected;
}


void Oscillator::expectedEnergy_diff(Vec x, Vec x_re_bar, Vec x_im_bar, double obj_bar) {

  int dimmat;
  MatGetSize(NumberOP, &dimmat, NULL);
  double num_diag;

  /* Derivative of projective measure */
  for (int i=0; i<dimmat; i++) {
    MatGetValue(NumberOP, i, i, &num_diag);
    int idx_diag = i * dimmat + i;
    double val = num_diag * obj_bar;
    VecSetValues(x_re_bar, 1, &idx_diag, &val, ADD_VALUES);
  }
  VecAssemblyBegin(x_re_bar); VecAssemblyEnd(x_re_bar);
}


void Oscillator::population(Vec x, std::vector<double> *pop) {

  int dimN = dim_preOsc * nlevels * dim_postOsc;

  assert ((*pop).size() == nlevels);

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
        int diagID = rhoID * dimN + rhoID;                // Position in vectorized rho
        double val;
        VecGetValues(x, 1, &diagID, &val);
        sum += val;
      }
    }
    (*pop)[i] = sum;
  } 
}