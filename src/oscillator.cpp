#include "oscillator.hpp"

Oscillator::Oscillator(){
  nlevels = 0;
  nparams = 0;
  param_Re = NULL;
  param_Im = NULL;
  Tfinal = 0;
  basisfunctions = NULL;
}

Oscillator::Oscillator(int nlevels_, int nbasis_, std::vector<double> carrier_freq_, double Tfinal_){
  nlevels = nlevels_;
  Tfinal = Tfinal_;
  basisfunctions = new Bspline(nbasis_, Tfinal_, carrier_freq_);

  /* Number of parameters per real and imaginary oscillator */
  // nparam = 2 * nbasis_ * carrier_freq_.size();
  nparams = 2 * nbasis_ * carrier_freq_.size();

  if (nparams>0) {
    param_Re = new double[nparams/2];
    param_Im = new double[nparams/2];
    for (int i=0; i<nparams/2; i++) {
      param_Re[i] = 0.0;
      param_Im[i] = 0.0;
    }
  }
}

Oscillator::~Oscillator(){
  if (nparams>0) {
    delete [] param_Re;
    delete [] param_Im;
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
  int j=0;
  for (int i=0; i<nparams/2; i++) {
    param_Re[i] = x[j]; 
    j++;
  }
  for (int i=0; i<nparams/2; i++) {
    param_Im[i] = x[j]; 
    j++;
  }
}

void Oscillator::getParams(double* x){
  int j=0;
  for (int i=0; i<nparams/2; i++) {
    x[j] = param_Re[i];
    j++;
  }
  for (int i=0; i<nparams/2; i++) {
    x[j] = param_Im[i];
    j++;
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
    *Re_ptr = basisfunctions->evaluate(t, param_Re);
    *Im_ptr = basisfunctions->evaluate(t, param_Im);
  }

  return 0;
}


int Oscillator::evalDerivative(double t, double* dRedp, double* dImdp) {

  if ( t > Tfinal ){
    printf("WARNING: accessing spline derivative outside of [0,T]. Returning 0.0\n");
    for (int i = 0; i < nparams/2; i++) {
      dRedp[i] = 0.0;
      dRedp[i] = 0.0;
    }
  } else {
      double Rebar = 1.0;
      double Imbar = 1.0;
      basisfunctions->derivative(t, dRedp, Rebar);
      basisfunctions->derivative(t, dImdp, Imbar);
  }

  return 0;
}
