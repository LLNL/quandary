#include <stdio.h>
#include "bspline.hpp"
#include <fstream>
#include <iomanip>
#include <petscmat.h>
#include <vector>

#pragma once

using namespace std;

class Oscillator {
  protected:
    int nlevels;  // Number of levels for this the oscillator 
    int nparam;   // Number of control parameters for each real and imaginary part
    double* param_Re; // parameters of real part of the control
    double* param_Im; // parameters of imaginary part of the control
    double Tfinal;               // final time
    Bspline *basisfunctions;     // Bspline basis function for control discretization 

  public:
    Oscillator();
    Oscillator(int nlevels_, int nbasis_, std::vector<double> carrier_freq_, double Tfinal_);
    virtual ~Oscillator();

    /* Return the constants */
    int getNParam() { return nparam; };
    int getNLevels() { return nlevels; };

    /* Get real and imaginary part of parameters.  */
    double* getParamsRe();
    double* getParamsIm();

    /* Print the control functions for each t \in [0,ntime*dt] */
    void flushControl(int ntime, double dt, const char* filename);

    /* Compute lowering operator a_k = I_n1 \kron ... \kron a^(nk) \kron ... \kron I_nQ */
    int createLoweringOP(int dim_prekron, int dim_postkron, Mat* loweringOP);

    /* Compute number operator N_k = a_k^T a_k */
    int createNumberOP(int dim_prekron, int dim_postcron, Mat* numberOP);

    /* Evaluates real and imaginary control function at time t */
    int evalControl(double t, double* Re_ptr, double* Im_ptr);

    /* Compute derivatives of the Re and Im control function wrt the parameters */
    int evalDerivative(double t, double* dRedp, double* dImdp);
};



