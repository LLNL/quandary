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
#include "data.hpp"
#include "UDEmodel.hpp"
#pragma once

class Learning {

  int mpirank_world;
  bool quietmode;
  int dim;              // Dimension of full vectorized system: N^2 for Lindblad, N for Schroedinger, or -1 if not learning.
  int dim_rho;               // Dimension of Hilbertspace = N
  LindbladType lindbladtype; // Switch for Lindblad vs Schroedinger solver
  UDEmodelType UDEmodel; // Switch type of learning model: hamiltonian, lindblad, or both

  HamiltonianModel* hamiltonian_model;  // Basis matrices for Hamiltonian term
  LindbladModel* lindblad_model;     // Basis matrices for Lindblad term 
  std::vector<double> learnparamsH;  // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsL; // Learnable parameters for Lindblad (first all real, then all imaginary parts)
  
  int nparams;            /* Total Number of learnable paramters*/
  double loss_integral;   /* Running cost for Loss function */
  Vec aux2;               /* Auxiliary state to eval loss */

  double loss_scaling_factor; /* Scaling the loss function value by this umber. Default=1.0 */

  public: 
    double current_err;
    Data* data;       /* Stores the data */

  public: 
    Learning(std::vector<int> nlevels, LindbladType lindbladtype_, UDEmodelType UDEmodel_, std::vector<std::string>& learninit_str, Data* data, std::default_random_engine rand_engine, bool quietmode, double loss_scaling_factor);
    ~Learning();

    void resetLoss(){ loss_integral = 0.0; };
    double getLoss() { return loss_integral; };

    /* Get total number of learnable parameters */
    int getNParams(){ return nparams; };
    int getNParamsHamiltonian(){ return learnparamsH.size();};
    int getNParamsLindblad(){ return learnparamsL.size(); };

    /* Initialize learnable parameters. */
    void initLearnParams(std::vector<std::string> learninit_str, std::default_random_engine rand_engine);

    /* Applies UDE terms to input state (u,v) */
    void applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout);
    /* Adjoint gradient: Sets (uout,vout) = dFWD^T *(u,v) */
    void applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout);

    /* Reduced gradient for Hamiltonian part: 
       Sets grad += alpha * (dRHS(u,v)/dgamma)^T *(ubar, vbar) */
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);

    /* Assemble and view the learned operator. */
    void writeOperators(std::string datadir);

    /* Copy optimization variable x into learnable parameter storage */
    void setLearnParams(const Vec x);
    /* Copy learnable parameters from storage into optimization variable x */
    void getLearnParams(double* x);

    /* Add to loss. Note: pulse_num is the global pulse number . */
    void addToLoss(double time, Vec x, int pulse_num);
    void addToLoss_diff(double time, Vec xbar, Vec xprimal, int pulse_num, double Jbar_loss);
};

