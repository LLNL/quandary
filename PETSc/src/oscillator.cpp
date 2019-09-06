#include "oscillator.hpp"

Oscillator::Oscillator(){}
Oscillator::~Oscillator(){}

int Oscillator::dumpControl(double tfinal, double dt){

  this->dumpControl(tfinal, dt, std::cout);
    return 0;
}


void Oscillator::dumpControl(double tfinal, double dt, std::ostream &output){

  int N = tfinal / dt;
  MultiVector controlout(N, 3, 0.0);

  /* Evaluate control at all time steps */
  for (int i=0; i < N; i++){
    double t = (double) i * dt;
    controlout(i,0) = i*dt;
    getControl(t,  &controlout(i,1), &controlout(i,2));
  }
  controlout.dump(output);
}

void Oscillator::dumpControl(double tfinal, double dt, std::string filename){
  ofstream file;
  file.open(filename.c_str());
  file << setprecision(20);
  this->dumpControl(tfinal, dt, file);
  file << endl;
  file.close();
}

SplineOscillator::SplineOscillator() {
  nbasis = 0;
  Tfinal = 0;
  basisfunctions = NULL;
  param_Re = NULL;
  param_Im = NULL;
}


SplineOscillator::SplineOscillator(int nbasis_, double Tfinal_){
  Tfinal = Tfinal_;
  nbasis = nbasis_;
  basisfunctions = new Bspline(nbasis_, Tfinal_);
  param_Re = new Vector(nbasis_, 0.0);
  param_Im = new Vector(nbasis_, 0.0);

  /* Set some initial parameters */
  for (int i=0; i<nbasis; i++){
    (*param_Re)(i) = pow(-1., i+1); //alternate 1 and -1
    (*param_Im)(i) = pow(-1., i+1); //alternate 1 and -1
  }
}

SplineOscillator::~SplineOscillator(){
  if (nbasis > 0){
    delete basisfunctions;
    delete param_Re;
    delete param_Im;
  }
}

int SplineOscillator::getControl(double t, double* Re_ptr, double* Im_ptr){

  if ( t > Tfinal ){
    printf("WARNING: accessing spline outside of [0,T]. Returning 0.0\n");
    *Re_ptr = 0.0;
    *Im_ptr = 0.0;
  } else {
    /* Evaluate the spline at time t */
    *Re_ptr = basisfunctions->evaluate(t, param_Re->GetData());
    *Im_ptr = basisfunctions->evaluate(t, param_Im->GetData());
  }

  return 0;
}

