#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
#include <fstream>
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


  double data_dtAWG;                /* Sample rate of AWG data (default 4ns) */
  int data_ntime;                   /* Number of data points in time */
  std::vector<Vec> data;            /* List of all data point (rho_data) at each data_dtAWG */

  Vec aux;    // Auxiliary vector to perform matvecs

  int mpirank_world;
  bool quietmode;

  public: 
    Learning(std::vector<int>&nlevels, LindbladType lindbladtype_, std::vector<std::string>& learninit_str, std::string data_name, double data_dtAWG_, int data_ntime, std::default_random_engine rand_engine, bool quietmode);
    ~Learning();

    /* Load data from file */
    void loadData(std::string data_name, double data_dtAWG, int data_ntime);

    /* Create generalized Gellman matrices, multiplied by (-i) and shifted s.t. G_00=0. Returns number of basis elements */
    int setupGellmannBasis(int dim_rho, int dim, LindbladType lindbladtype);

    /* Initialize learnable parameters */
    void initLearnableParams(std::vector<std::string> learninit_str, int nbasis, int dim_rho, std::default_random_engine rand_engine);

    /* Applies Learning operator to input state (u,v) */
    void applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout);

    /* Adjoint gradient: Sets (uout,vout) = dFWD^T *(u,v) */
    void applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout);

    /* Reduced gradient: Sets grad += alpha * (dRHS(u,v)/dgamma)^T *(ubar, vbar) */
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);

    /* Get size of the basis: N^2-1, or 0 if no learning */
    int getNBasis(){ return nbasis; };

    /* Get data trajectory element */
    Vec getData(int id) {assert(data.size()>id); return data[id];};

    /* Get number of data elements */
    int getNData(){ return data.size(); };

    /* Assemble the learned operator. Allocates the (dense!) return matrix, which hence must be destroyed after usage. */
    void getLearnOperator(Mat* A, Mat* B);

    /* Pass learnable parameters to storage learnparamsH_A and learnparamsH_B*/
    void setLearnParams(const Vec x);

    /* Copy learnable parameters from storage into x */
    void getLearnParams(double* x);
};

