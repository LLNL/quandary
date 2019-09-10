#include <stdio.h>
#include "vector.hpp"
#include "bspline.hpp"
#include <fstream>
#include <iomanip>
#pragma once

using namespace std;

/*
 * Abstract base class for oscillators
 */
class Oscillator {

  public:
    Oscillator();
    virtual ~Oscillator();

    /* Evaluates real and imaginary control function at time t */
    virtual int getControl(double t, double* Re_ptr, double* Im_ptr) = 0;

    /* Return pointers to the control parameters */
    virtual int getParams(double* paramsRe, double* paramsIm) = 0;

    /* Print the control functions for each t \in [0,tfinal] */
    virtual int dumpControl(double tfinal, double dt);
    virtual void dumpControl(double tfinal, double dt, std::ostream &output);
    virtual void dumpControl(double tfinal, double dt, std::string filename);
};


/* 
 * Implements oscillators that are discretized by spline basis functions
 */
class SplineOscillator : public Oscillator {
    double Tfinal;               // final time
    int nbasis;               // Dimension of control discretization
    Bspline *basisfunctions;  // Bspline basis function for control discretization 
    Vector* param_Re;          // parameters of real part of the control
    Vector* param_Im;          // parameters of imaginary part of the control

  public:
    SplineOscillator();
    SplineOscillator(int nbasis, double Tfinal_);
    ~SplineOscillator();

    /* Evaluates the real and imaginare spline functions at time t, using current spline parameters */
    virtual int getControl(double t, double* Re_ptr, double* Im_ptr);

    /* Returns pointers to the real and imaginary control parameters */
    virtual int getParams(double* paramsRe, double* paramsIm);
};


class FunctionOscillator : public Oscillator {

  double (*F)(double t, double freq);  // function pointer to Re(control function)
  double (*G)(double t, double freq);  // function pointer to Im(control function)
  double omegaF;    // Optim parameter: Frequency for (*F)
  double omegaG;    // Optim parameter: Frequency for (*G)

  public:
    FunctionOscillator();
    FunctionOscillator( double omegaF_, double (*F_)(double, double), double omegaG_, double (*G_)(double, double) );
    ~FunctionOscillator();

    /* Evaluates the control functions at time t */
    virtual int getControl(double t, double* Re_ptr, double* Im_ptr);

    /* Returns pointers to the real and imaginary control frequencies */
    virtual int getParams(double* paramsRe, double* paramsIm);
};