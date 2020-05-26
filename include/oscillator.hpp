#include <stdio.h>
#include "bspline.hpp"
#include <fstream>
#include <iostream> 
#include <iomanip>
#include <petscmat.h>
#include <vector>
#include <assert.h>

#pragma once

using namespace std;

class Oscillator {
  protected:
    int nlevels;                   // Number of levels for this the oscillator 
    double ground_freq;            // Ground frequency of this oscillator
    std::vector<double> params;    // control parameters 
    double Tfinal;                 // final time
    ControlBasis *basisfunctions;  // Control discretization using Bsplines + carrier waves
    Mat NumberOP;                  // Stores the number operator
    Mat LoweringOP;                // Stores the lowering operator
    int dim_preOsc;                // Dimension of coupled subsystems preceding this oscillator
    int dim_postOsc;               // Dimension of coupled subsystem following this oscillator

    Mat zeromat;

    int mpirank_petsc;

  public:
    Oscillator();
    Oscillator(int id, std::vector<int> nlevels_all_, int nbasis_, double ground_freq_, std::vector<double> carrier_freq_, double Tfinal_);
    virtual ~Oscillator();

    /* Return the constants */
    int getNParams() { return params.size(); };
    int getNLevels() { return nlevels; };

    /* Copy x to real and imaginary part of parameter */
    void setParams(const double* x);
    /* Copy real and imaginary part of parameters into x */
    void getParams(double* x);

    /* Print the control functions for each t \in [0,ntime*dt] */
    void flushControl(int ntime, double dt, const char* filename);

    /* Compute lowering operator a_k = I_n1 \kron ... \kron a^(nk) \kron ... \kron I_nQ */
    int createLoweringOP(int dim_prekron, int dim_postkron, Mat* loweringOP);
    /* Returns the lowering operator, unless dummy is true, then return a zero matrix */
    Mat getLoweringOP(bool dummy);

    /* Compute number operator N_k = a_k^T a_k */
    int createNumberOP(int dim_prekron, int dim_postcron, Mat* numberOP);
    /* Returns the number operator, unless dummy is true, then return a zero matrix */
    Mat getNumberOP(bool dummy);

    /* Evaluates rotating frame control functions Re = p(t), Im = q(t) */
    int evalControl(double t, double* Re_ptr, double* Im_ptr);

    /* Evaluates Lab-frame control function f(t) */
    int evalControl_Labframe(double t, double* f_ptr);

    /* Compute derivatives of the Re and Im control function wrt the parameters */
    int evalDerivative(double t, double* dRedp, double* dImdp);

    /* Return expected value of projective measure in basis |m> */
    double expectedEnergy(Vec x);
    void expectedEnergy_diff(Vec x, Vec x_bar, double obj_bar);

   void population(Vec x, std::vector<double> *pop); 
};



