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



Normal_GaussHermit::Normal_GaussHermit(std::vector<std::string>& inputstring) : Sampler(inputstring) {
  if (inputstring.size() < 3 ) {
    printf("Error: Input for config option 'optim_robust' invalid\n");
    exit(1);
  }

  /* Get mean and standard deviation */
  mu    = atof(inputstring[2].c_str());  // GHz!
  sigma = atof(inputstring[3].c_str());  // GHz!

  if (nsamples != 5 && nsamples !=10 && nsamples !=15 && nsamples !=20) {
    printf("WARNING: Gauss-Hermite quadrature points only available for nsamples 5,10,15, or 20. Switching to nsamples=10 now.\n");
    nsamples = 10;
  }

  /* Set the samples and double xweights as N(mu,sigma) */
  // https://jblevins.org/notes/quadrature
  double *abs_ptr, *w_ptr;
  if      (nsamples == 5)  { abs_ptr = abs5;  w_ptr = w5; } 
  else if (nsamples == 10) { abs_ptr = abs10; w_ptr = w10; } 
  else if (nsamples == 15) { abs_ptr = abs15; w_ptr = w15; } 
  else if (nsamples == 20) { abs_ptr = abs20; w_ptr = w20; } 

  for (int i=0; i<nsamples; i++) {
    double val = w_ptr[1] / sqrt(M_PI) ;
    samples.push_back( mu + sqrt(2) * sigma * abs_ptr[i]); 
    coeffs.push_back( w_ptr[i] / sqrt(M_PI) );
  }

  /* Set coefficients according to GaussHermite quadrature w_i / sqrt(pi) */
 
  for (int i=0; i<nsamples; i++){
    printf("%d, x=%f, w=%f\n",i, samples[i], coeffs[i]);
  }
  exit(1);

}

double Normal_GaussHermit::evalPDF(double x) {
  return 1./(sqrt(2.*M_PI)*sigma) * exp( -1./2. * pow( (x - mu) / sigma , 2) );
}

