#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#pragma once

/* Base class for Robust optimization sampler */
class Sampler {
  protected:
    int nsamples;

  public:
    std::vector<double> samples;
    std::vector<double> weights;

    Sampler();
    ~Sampler();

};

class Uniform : public Sampler {
  protected:
    double pstart, pstop, deltap;

  public:
    Uniform(std::vector<std::string>& inputstring);
    ~Uniform();
};