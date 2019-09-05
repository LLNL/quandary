#include "oscillator.hpp"
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
    Mat H;                  // Realvalued, vectorized Hamiltonian operator

  public:
    Hamiltonian();
    ~Hamiltonian();

    /* Return dimension of vectorized system */
    int getDim();

    /* Sets the variables and allocates Re, Im, H */
    virtual int initialize(int nlevels_, int noscillators_, Oscillator** oscil_vec_);

    /* Apply the Hamiltonian operator */
    virtual int apply(double t);
    /* Access the Hamiltonian */
    Mat getH();
};

/*
 * Hamiltonian with two oscillators 
 */
class TwoOscilHam : public Hamiltonian {

  Mat A1, A2;  // Building blocks for real part of Hamiltonian
  Mat B1, B2;  // Building blocks for imaginary part of Hamiltonian
  Mat Hd;      // Constant part of Hamiltonian matrix ("drift Hamiltonian")

  public:
    TwoOscilHam(int nlevels_, Oscillator** oscil_vec_); 
    ~TwoOscilHam(); 

    /* Set up the building block matrices */ 
    int initialize();

    /* Helper function for constructing building blocks */
    int BuildingBlock(Mat C, int sign, int k, int m);

    /* Apply the Hamiltonian operator */
    virtual int apply(double t);
 
};