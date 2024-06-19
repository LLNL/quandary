#include "defs.hpp"
#include <petscts.h>
#include <vector>
#pragma once


class Learning {

  int dim;              // Dimension of full vectorized system: N^2 for Lindblad
  int dim_rho;               // Dimension of Hilbertspace = N
  LindbladType lindbladtype; // Switch for Lindblad vs Schroedinger solver



  public: 
    Learning(const int dim_, LindbladType lindbladtype_);
    ~Learning();

  std::vector<Mat> GellmannMats_A;      // Real(-i*GellmannMatx), for the generalized & shifted Gellmann matrices
  std::vector<Mat> GellmannMats_B;      // Imag(-i*GellmannMatx), for the generalized & shifted Gellmann matrices
  std::vector<double> learnparamsH_A; // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsH_B; // Learnable parameters for Hamiltonian
};