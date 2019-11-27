#include "oscillator.hpp"

Oscillator::Oscillator(){
  nparam = 0;
  nlevels = 0;
}
Oscillator::~Oscillator(){}

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



SplineOscillator::SplineOscillator() {
  Tfinal = 0;
  basisfunctions = NULL;
  param_Re = NULL;
  param_Im = NULL;
}


SplineOscillator::SplineOscillator(int nlevels_, int nbasis_, double Tfinal_){
  Tfinal = Tfinal_;
  nlevels = nlevels_;
  nparam = nbasis_;  // nparam for Re and nparam for Im ! 
  basisfunctions = new Bspline(nparam, Tfinal_);
  param_Re = new double[nparam];
  param_Im = new double[nparam];

  /* Set some initial parameters */
  for (int i=0; i<nparam; i++){
    param_Re[i] = pow(-1., i); //alternate 1 and -1
    param_Im[i] = pow(-1., i+1); //alternate 1 and -1
  }
}

SplineOscillator::~SplineOscillator(){
  if (nparam > 0){
    delete basisfunctions;
    delete [] param_Re;
    delete [] param_Im;
  }
}

int SplineOscillator::evalControl(double t, double* Re_ptr, double* Im_ptr){

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

int SplineOscillator::getParams(double* paramsRe, double* paramsIm) {

  paramsRe = param_Re;
  paramsIm = param_Im;

  return 0;
}


int SplineOscillator::evalDerivative(double t, double* dRedp, double* dImdp) {

  if ( t > Tfinal ){
    printf("WARNING: accessing spline derivative outside of [0,T]. Returning 0.0\n");
    for (int i = 0; i < nparam; i++) {
      dRedp[i] = 0.0;
      dRedp[i] = 0.0;
    }
  } else {
      double Rebar = 1.0;
      double Imbar = 1.0;
      basisfunctions->derivative(t, param_Re, Rebar, dRedp);
      basisfunctions->derivative(t, param_Im, Imbar, dImdp);
  }

  return 0;
}

int SplineOscillator::updateParams(double stepsize, double* directionRe, double* directionIm){

  for (int i=0; i<nparam; i++) {
    param_Re[i] += stepsize * directionRe[i];
    param_Im[i] += stepsize * directionIm[i];
  }

  return 0;
}

FunctionOscillator::FunctionOscillator() {
  F = NULL;
  G = NULL;
  dFdp = NULL;
  dGdp = NULL;
  param_Re = NULL;
  param_Im = NULL;
}

FunctionOscillator::FunctionOscillator( int nlevels_,
        double omegaF_, 
        double (*F_)(double, double), 
        double (*dFdp_) (double, double, double), 
        double omegaG_, 
        double (*G_)(double, double), 
        double (*dGdp_) (double, double, double) ) {
  nlevels = nlevels_;
  F = F_;
  G = G_;
  dFdp = dFdp_;
  dGdp = dGdp_;
  nparam = 1;  // one for F (Re) one for G (Im)
  param_Re = new double[nparam];
  param_Im = new double[nparam];
  param_Re[0] = omegaF_;
  param_Im[0] = omegaG_;
}

FunctionOscillator::~FunctionOscillator(){
  delete [] param_Re;
  delete [] param_Im;
}

int FunctionOscillator::evalControl(double t, double* Re_ptr, double* Im_ptr){

  double Re = 0.0;
  double Im = 0.0;

  /* Evaluate F and G, if set */
  if (F != NULL) Re = (*F)(t, param_Re[0]);
  if (G != NULL) Im = (*G)(t, param_Im[0]);

  *Re_ptr = Re;
  *Im_ptr = Im;

  return 0;
}


int FunctionOscillator::evalDerivative(double t, double* dRedp, double* dImdp) {

  double Fbar = 1.0;
  double Gbar = 1.0;
  /* Evaluate derivative functions */
  if (F != NULL) *dRedp = (*dFdp)(t, param_Re[0], Fbar);
  else  *dRedp = 0.0;
  if (G != NULL) *dImdp = (*dGdp)(t, param_Im[0], Gbar);
  else *dImdp = 0.0;

  return 0;
}

int FunctionOscillator::getParams(double* paramsRe, double* paramsIm) {
  *paramsRe = param_Re[0];
  *paramsIm = param_Im[0];

  return 0;
}

int FunctionOscillator::updateParams(double stepsize, double* directionRe, double* directionIm) {

  if (F != NULL) param_Re[0] += stepsize * (directionRe[0]);
  if (G != NULL) param_Im[0] += stepsize * (directionIm[0]);
  // printf("f=%1.8f, g=%1.8f\n", param_Re[0], param_Im[0]);

  return 0;
}