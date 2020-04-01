#include "oscillator.hpp"

Oscillator::Oscillator(){
  nlevels = 0;
  Tfinal = 0;
  basisfunctions = NULL;
}

Oscillator::Oscillator(int nlevels_, int nbasis_, std::vector<double> carrier_freq_, double Tfinal_){
  nlevels = nlevels_;
  Tfinal = Tfinal_;
  basisfunctions = new ControlBasis(nbasis_, Tfinal_, carrier_freq_);
  printf("Creating oscillator with %d levels\n", nlevels);

  /* Initialize control parameters */
  int nparam = 2 * nbasis_ * carrier_freq_.size();
  for (int i=0; i<nparam; i++) {
    params.push_back(0.0);
  }
}

Oscillator::~Oscillator(){
  if (params.size() > 0) {
    delete basisfunctions;
  }
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
