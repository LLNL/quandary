#include <stdio.h>
#include "vector.hpp"
#include "bspline.hpp"
#pragma once

/*
 * Abstract base class for oscillators
 */
class Oscillator {

  public:
    Oscillator();
    virtual ~Oscillator();

    /* Evaluates real and imaginary control function at time t */
    virtual int getControl(double t, double* Re_ptr, double* Im_ptr) = 0;

    /* Print Control */
    virtual int dumpControl() = 0;
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

    virtual int getControl(double t, double* Re_ptr, double* Im_ptr);

    virtual int dumpControl();
};