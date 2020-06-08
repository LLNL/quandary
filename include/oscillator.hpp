#include <stdio.h>
#include "bspline.hpp"
#include <fstream>
#include <iostream> 
#include <iomanip>
#include <petscmat.h>
#include <vector>
#include <assert.h>
#include "util.hpp"

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

    Mat zeromat;                   // auxiliary matrix with zero entries
    int mpirank_petsc;             // rank of Petsc's communicator

  public:
    Oscillator();
    Oscillator(int id, std::vector<int> nlevels_all_, int nbasis_, double ground_freq_, std::vector<double> carrier_freq_, double Tfinal_);
    virtual ~Oscillator();

    /* Return the constants */
    int getNParams() { return params.size(); };
    int getNLevels() { return nlevels; };

    /* Copy x into the control parameter vector */
    void setParams(const double* x);

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
    /* Compute derivatives of the p(t) and q(t) control function wrt the parameters */
    int evalControl_diff(double t, double* dRedp, double* dImdp);

    /* Evaluates Lab-frame control function f(t) */
    int evalControl_Labframe(double t, double* f_ptr);

    /* Return expected value of projective measure in basis |m> */
    double expectedEnergy(Vec x);
    /* Derivative of expected alrue computation */
    void expectedEnergy_diff(Vec x, Vec x_bar, double obj_bar);

    /* Compute population (=diagonal elements) for this oscillators reduced system */
    void population(Vec x, std::vector<double> *pop); 
};



