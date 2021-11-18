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

struct PiPulse {
  std::vector<double> tstart; 
  std::vector<double> tstop;
  std::vector<double> amp;
};

class Oscillator {
  protected:
    int nlevels;                   // Number of levels for this the oscillator 
    double ground_freq;            // Fundamental transition frequency of this oscillator
    double selfkerr;               // Self-kerr frequency $\xi_k$. Multiplies ak^d ak^d ak ak

    double detuning_freq;              // Detuning frequency (rad/time) for this oscillator. Multiplies ak^d ak in rotating frame: detuning = ground_freq - rotational_freq
    double decay_time;              // Time of decay collapse operations 
    double dephase_time;           // Time of dephasing dephasing collapse operations 

    std::vector<double> params;    // control parameters 
    double Tfinal;                 // final time
    ControlBasis *basisfunctions;  // Control discretization using Bsplines + carrier waves

    int mpirank_petsc;             // rank of Petsc's communicator

  public:
    PiPulse pipulse;  // Store a dummy pipulse that does nothing
    int dim_preOsc;                // Dimension of coupled subsystems preceding this oscillator
    int dim_postOsc;               // Dimension of coupled subsystem following this oscillator


  public:
    Oscillator();
    Oscillator(int id, std::vector<int> nlevels_all_, int nbasis_, double ground_freq_, double selfkerr_, double rotational_freq_, double decay_time_, double dephase_time_, std::vector<double> carrier_freq_, double Tfinal_);
    virtual ~Oscillator();

    /* Return the constants */
    int getNParams() { return params.size(); };
    int getNLevels() { return nlevels; };
    int getNSplines() { return basisfunctions->getNSplines(); };
    int getNCarrierwaves() {return basisfunctions->getNCarrierwaves(); };
    double getSelfkerr() { return selfkerr; }; 
    double getDetuning() { return detuning_freq; }; 
    double getDecayTime() {return decay_time; };
    double getDephaseTime() {return dephase_time; };

    /* Copy x into the control parameter vector */
    void setParams(const double* x);

    /* Evaluates rotating frame control functions Re = p(t), Im = q(t) */
    int evalControl(const double t, double* Re_ptr, double* Im_ptr);
    /* Compute derivatives of the p(t) and q(t) control function wrt the parameters */
    int evalControl_diff(const double t, double* dRedp, double* dImdp);

    /* Evaluates Lab-frame control function f(t) */
    int evalControl_Labframe(const double t, double* f_ptr);

    /* Return expected value of projective measure in basis |m> */
    double expectedEnergy(const Vec x);
    /* Derivative of expected alrue computation */
    void expectedEnergy_diff(const Vec x, Vec x_bar, const double obj_bar);

    /* Compute population (=diagonal elements) for this oscillators reduced system */
    void population(const Vec x, std::vector<double> &pop); 
};



