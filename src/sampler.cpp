#include "sampler.hpp"

/* Abstract base class for Robust optimization sampler */
Sampler::Sampler() {
  nsamples = 0;
}

Sampler::~Sampler(){}

Uniform::Uniform(std::vector<std::string>& inputstring) : Sampler(){

  /* Get the range and spacing for uniform distribution */
  pstart = atof(inputstring[1].c_str());  // GHz! 
  pstop  = atof(inputstring[2].c_str());  // GHz! 
  deltap = atof(inputstring[3].c_str());  // GHz! 

  /* Get number of samples */
  nsamples = (int) ( (pstop - pstart) / deltap );
  nsamples +=1;

  /* Set the samples uniformly */
  for (int i=0; i < nsamples; i++) {
    samples.push_back( (pstart + i*deltap) * 2. * M_PI );
    weights.push_back( 1./nsamples );
  }
}

Uniform::~Uniform(){}



Normal::Normal(std::vector<std::string>& inputstring){
  /* Get mean and standard deviation */
  mu    = atof(inputstring[1].c_str());  // GHz!
  sigma = atof(inputstring[2].c_str());  // GHz!


  /* Set the samples and weights as N(mu,sigma) */
  // https://jblevins.org/notes/quadrature
}