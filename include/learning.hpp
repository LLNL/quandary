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

/* Generalized Gellman matrices, multiplied by (-i) and shifted s.t. G_00=0 */
class GenGellmannBasis {

  int dim_rho;   /* Dimension of the Hilbertspace (N)*/
  int dim;       /* N (if Schroedinger solver) or N^2 (if Lindblad) */
  int nbasis;    /* Total number of basis matrices */
  int nsigma;    /* =N*(N-1)/2. Here, number of offdiagonal real (or imag) basis matrices. */

  public:
    std::vector<Mat> BasisMats_A;  // Real(-i*GellmannMatx), for the generalized & shifted Gellmann matrices
    std::vector<Mat> BasisMats_B;  // Imag(-i*GellmannMatx), for the generalized & shifted Gellmann matrices

  public:
    GenGellmannBasis(int dim_rho_, LindbladType lindbladtype_);
    ~GenGellmannBasis();

    int getNBasis(){return nbasis;};
    int getNBasis_A(){return BasisMats_A.size();};
    int getNBasis_B(){return BasisMats_B.size();};
};

class Learning {

  int dim;              // Dimension of full vectorized system: N^2 for Lindblad, N for Schroedinger, or -1 if not learning.
  int dim_rho;               // Dimension of Hilbertspace = N
  LindbladType lindbladtype; // Switch for Lindblad vs Schroedinger solver

  GenGellmannBasis* hamiltonian_basis; 
  std::vector<double> learnparamsH_A; // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsH_B; // Learnable parameters for Hamiltonian
  
  int nparams;           /* Total Number of learnable paramters*/

  double data_dtAWG;     /* Sample rate of AWG data (default 4ns) */
  int data_ntime;        /* Number of data points in time */
  int loss_every_k;      /* Add to loss at every k-th timestep */
  std::vector<Vec> data; /* List of all data point (rho_data) at each data_dtAWG */

  Vec aux;     // Auxiliary vector to perform matvecs on Re(x) or Im(x)
  Vec aux2;    // Auxiliary vector to perform matvecs on x

  int mpirank_world;
  bool quietmode;

  double loss_integral;   /* Running cost for Loss function */

  public: 
    Learning(std::vector<int>&nlevels, LindbladType lindbladtype_, std::vector<std::string>& learninit_str, std::string data_name, double data_dtAWG_, int data_ntime, int loss_every_k, std::default_random_engine rand_engine, bool quietmode);
    ~Learning();

    void resetLoss(){ loss_integral = 0.0; };
    double getLoss() { return loss_integral; };

    /* Get total number of learnable parameters */
    int getNParams(){ return nparams; };

    /* Initialize learnable parameters. Return total number of params. */
    int initLearnableParams(std::vector<std::string> learninit_str, std::default_random_engine rand_engine);

    /* Applies Learning operator to input state (u,v) */
    void applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout);

    /* Adjoint gradient: Sets (uout,vout) = dFWD^T *(u,v) */
    void applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout);

    /* Reduced gradient: Sets grad += alpha * (dRHS(u,v)/dgamma)^T *(ubar, vbar) */
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);

    /* Load data from file */
    void loadData(std::string data_name, double data_dtAWG, int data_ntime);

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

    /* Add to loss */
    void addToLoss(int timestepID, Vec x);
    void addToLoss_diff(int timestepID, Vec xbar, Vec xprimal, double Jbar_loss);
};

