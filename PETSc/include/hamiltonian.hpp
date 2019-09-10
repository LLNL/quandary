#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#pragma once

/* 
 * Abstract base class for Hamiltonian systems 
 */
class Hamiltonian{
  protected:
    int dim;                 // Dimension of vectorized system 
    int nlevels;             // Number of levels
    int noscillators;        // Number of oscillators 
    Oscillator** oscil_vec;  // Vector storing pointers to the oscillators

    Mat Re, Im;             // Real and imaginary part of Hamiltonian operator
    Mat M;                  // Realvalued, vectorized Hamiltonian operator vec(-i(Hq-qH))

  public:
    /* Default constructor sets zero */
    Hamiltonian();
    /* This constructor sets the variables and allocates Re, Im and M */
    Hamiltonian(int nlevels_, int noscillators_, Oscillator** oscil_vec_);
    ~Hamiltonian();

    /* Return dimension of vectorized system */
    int getDim();

    /* 
     * Evaluate the exact solution at time t.
     * Inheriting classes should store exact solution in x and return true.  
     * Return false, if exact solution is not available.
     */
    virtual bool ExactSolution(double t, Vec x);

    /* 
     * Uses Re and Im to build the Hamiltonian operator vectorized M = vec(-i(Hq-qH)). 
     * M(0, 0) =  Re    M(0,1) = -Im
     * M(1, 0) =  Im    M(1,1) = Re
     * Both Re and Im should be set up in the inherited 'apply' routines. 
     */
    virtual int apply(double t);

    /* 
     * Set x to the initial condition 
     */
    virtual int initialCondition(Vec x) = 0;

    /* Access the Hamiltonian */
    Mat getM();

    /* 
     * Evaluate the objective function at time t and current solution x
     */
    virtual int evalObjective(double t, Vec x, double *objective_ptr);
};

/*
 * Hamiltonian with two oscillators 
 */
class TwoOscilHam : public Hamiltonian {

  Mat A1, A2;  // Building blocks for real part of Hamiltonian
  Mat B1, B2;  // Building blocks for imaginary part of Hamiltonian
  Mat Hd;      // Constant part of Hamiltonian matrix ("drift Hamiltonian")

  double* xi;   // xi = xi1, xi2, xi12

  public:
    TwoOscilHam();
    TwoOscilHam(int nlevels_, double* xi, Oscillator** oscil_vec_); 
    ~TwoOscilHam(); 

    /* Helper function for constructing building blocks */
    int BuildingBlock(Mat C, int sign, int k, int m);

    /* Set the initial condition (zero so far...) */
    int initialCondition(Vec x);

    /* Evaluate Re and Im of the Hamiltonian. Then calls the base-class apply routine to set up M. */
    virtual int apply(double t);

};


/* 
 * Ander's testcase with analytic solution 
 */
class AnalyticHam : public TwoOscilHam {
  
  public:
    AnalyticHam(double* xi_, Oscillator** oscil_vec_);
    ~AnalyticHam();

    /* Evaluate the exact solution at time t. */
    virtual bool ExactSolution(double t, Vec x);

    /* Set the initial condition (exact(0)) */
    int initialCondition(Vec x);
};

/* Real part for Oscillator 1 of analytic solution */
PetscScalar F1_analytic(PetscReal t,PetscReal freq);

/* Imaginary part for Oscillator 2 of analytic solution */
PetscScalar G2_analytic(PetscReal t,PetscReal freq);

