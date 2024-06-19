#include "defs.hpp"
#include <petscts.h>
#include <vector>
#pragma once


class Learning {

  int dim;              // Dimension of full vectorized system: N^2 for Lindblad
  int dim_rho;               // Dimension of Hilbertspace = N
  LindbladType lindbladtype; // Switch for Lindblad vs Schroedinger solver

  std::vector<double> learn_params_H; // Learnable parameters for Hamiltonian -> N^2-1 many
  std::vector<double> learn_params_C; // Learnable parameters for Collapse Operators -> N^2(N^2-1)/2 many

  std::vector<Mat> GellmannMats;

  public: 
    Learning(const int dim_, LindbladType lindbladtype_);
    ~Learning();
};