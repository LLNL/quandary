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
    std::vector<double> coeffs;

    Sampler();
    Sampler(std::vector<std::string>& inputstring);
    ~Sampler();

    double evalPDF(double x);
};

/* Uniform distribution, Trapez rule for quadrature of expected value integral */
class Uniform_Trapez : public Sampler {
  protected:
    double xstart, xstop, deltax;

  public:
    Uniform_Trapez(std::vector<std::string>& inputstring);
    ~Uniform_Trapez();

    /* Evaluate the probability density function */
    double evalPDF(double x);
};

class Normal : public Sampler {
  protected:
    double mu;     // Expected values
    double sigma;  // standard deviation

    public:
      Normal(std::vector<std::string>& inputstring);
      ~Normal();

      /* Evaluate the probability density function */
      double evalPDF(double x);
};
