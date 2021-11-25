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

class Normal_GaussHermit : public Sampler {
  protected:
    double mu;     // Expected values
    double sigma;  // standard deviation
    
    //
    // https://dlmf.nist.gov/3.5#v
    double abs5[5] = {-2.02018,-0.958572,0.0,0.958572,2.02018};
    double   w5[5] = {0.019953,0.39361,0.94530,0.39361,0.019953};
    double abs10[10] = {-3.43616,-2.53273,-1.75668,-1.03661,-0.342901,0.342901,1.03661,1.75668,2.53273,3.43616};
    double   w10[10] = {0.0000076404,0.001343,0.033874,0.24013,0.61086,0.61086,0.24013,0.033874,0.001343,0.0000076404};
    double abs15[15] = {-4.4999,-3.6699,-2.9671,-2.3257,-1.7199,-1.1361,-0.5650,0.0,0.5650,1.1361,1.7199,2.3257,2.9671,3.6699,4.4999};
    double   w15[15] = {0.0000000015224,0.0000010591,0.0001,0.0027780,0.030780,0.15848,0.41202,0.56410,0.41202,0.15848,0.030780,0.0027780,0.00010000,0.0000010591,0.0000000015224};
    double abs20[20] = {-5.3874,-4.6036,-3.9447,-3.3478,-2.7888,-2.2549,-1.7385,-1.2340,-0.7374,-0.2453,0.2453,0.7374,1.2340,1.7385,2.2549,2.7888,3.3478,3.9447,4.6036,5.3874};
    double   w20[20] = {0.00000000000022293,0.00000000043993,0.00000010860,0.0000078025,0.00022833,0.0032437,0.024810,0.10901,0.28667,0.46224,0.46224,0.28667,0.10901,0.024810,0.0032437,0.00022833,0.0000078025,0.00000010860,0.00000000043993,0.00000000000022293};


    public:
      Normal_GaussHermit(std::vector<std::string>& inputstring);
      ~Normal_GaussHermit();

      /* Evaluate the probability density function */
      double evalPDF(double x);
};
