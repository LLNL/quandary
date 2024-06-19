#include "util.hpp"
#include "defs.hpp"
#include <assert.h>
#include <petscts.h>
#include <vector>
#include<random>
#pragma once


class Learning {

  int dim;              // Dimension of full vectorized system: N^2 for Lindblad, N for Schroedinger, or -1 if not learning.
  int dim_rho;               // Dimension of Hilbertspace = N
  LindbladType lindbladtype; // Switch for Lindblad vs Schroedinger solver
  int nbasis;           // Number of basis elements (N^2-1, or 0 if not learning)

  std::vector<Mat> GellmannMats_A;      // Real(-i*GellmannMatx), for the generalized & shifted Gellmann matrices
  std::vector<Mat> GellmannMats_B;      // Imag(-i*GellmannMatx), for the generalized & shifted Gellmann matrices
  std::vector<double> learnparamsH_A; // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsH_B; // Learnable parameters for Hamiltonian

  Vec aux;    // Auxiliary vector to perform matvecs

  int mpirank_world;

  public: 
    Learning(std::vector<int>&nlevels, LindbladType lindbladtype_, std::vector<std::string>& learninit_str, std::default_random_engine rand_engine, bool quietmode);
    ~Learning();

    /* Applies Learning operator to input state (u,v) */
    void applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout);

    /* Get size of the basis: N^2-1, or 0 if no learning */
    int getNBasis(){ return nbasis; };
};