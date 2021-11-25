#include "sampler.hpp"

Sampler::Sampler(){
  nsamples = 0;
}
Sampler::Sampler(std::vector<std::string>& inputstring) {
  nsamples = atoi(inputstring[1].c_str());
}

Sampler::~Sampler(){}

double Sampler::evalPDF(double x) {return 0.0;}

Uniform_Trapez::Uniform_Trapez(std::vector<std::string>& inputstring) : Sampler(inputstring){
  if (inputstring.size() < 4 ) {
    printf("Error: Input for config option 'optim_robust' invalid\n");
    exit(1);
  }

  /* Get the range for uniform distribution */
  xstart = atof(inputstring[2].c_str());   
  xstop  = atof(inputstring[3].c_str());   
  deltax = (xstop - xstart) / (nsamples - 1); 

  /* Set the samples x_i uniformly */
  for (int i=0; i < nsamples; i++) {
    samples.push_back( xstart + i*deltax );
  }

  /* Set the coeffients according to trapez rule [ 1/2 1 1 ... 1 1 1/2 ] * p(x_i) * deltax */
  coeffs.push_back( 1./2. * evalPDF(samples[0]) * deltax);
  for (int i=1; i<nsamples-1; i++){
    coeffs.push_back( 1. * evalPDF(samples[i]) * deltax);
  }
  coeffs.push_back( 1./2. * evalPDF(samples[nsamples-1]) * deltax);

}

Uniform_Trapez::~Uniform_Trapez(){}
    
double Uniform_Trapez::evalPDF(double x){
  return 1./(xstop - xstart);
}



Normal::Normal(std::vector<std::string>& inputstring) : Sampler(inputstring) {
  if (inputstring.size() < 3 ) {
    printf("Error: Input for config option 'optim_robust' invalid\n");
    exit(1);
  }

  /* Get mean and standard deviation */
  mu    = atof(inputstring[2].c_str());  // GHz!
  sigma = atof(inputstring[3].c_str());  // GHz!


  /* Set the samples and double xweights as N(mu,sigma) */
  // https://jblevins.org/notes/quadrature
}

double Normal::evalPDF(double x) {
  // TODO.
  return 0.0;
}

