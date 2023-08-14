#include "defs.hpp"
#include <math.h>
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#include <controlbasis.hpp>

#pragma once

class PythonInterface{

  protected:

    LindbladType lindbladtype;            // Storing whether Lindblas solver or Schroedinger solver
    int dim_rho;                          // Dimension of the Hilbertspace. N!
    std::vector<int>ncontrol_real;
    std::vector<int>ncontrol_imag;
    std::string hamiltonian_file; // either 'none' or name of file to read Hamiltonian from 
    int mpirank_world;   // Rank of global communicator

	public:
    PythonInterface();
    PythonInterface(std::string hamiltonian_file_, LindbladType lindbladtype_, int dim_rho_);
    ~PythonInterface();

  /* Read the constant system Hamiltonian from file */
  // Hd must be REAL valued!
  void receiveHsys(Mat& Bd);

  /* Receive real and imaginary control operators from file */
  void receiveHc(int noscillators, std::vector<std::vector<Mat>>& Ac_vec, std::vector<std::vector<Mat>>& Bc_vec);
};
