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

  HamiltonianModel* hamiltonian_model;  // Parameterization of Hamiltonian
  LindbladModel* lindblad_model;        // Parameterization of Lindblad 
  std::vector<TransferModel*> transfer_model;   // Vector of Parameterization for control transfer, one for each oscillator
  std::vector<double> learnparamsH;  // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsL; // Learnable parameters for Lindblad (first all real, then all imaginary parts)
  std::vector<std::vector<double>> learnparamsT; // Learnable parameters for Transfer functions, one vector for each oscillator
  
  int nparams;            /* Total Number of learnable paramters*/
  double loss_integral;   /* Running cost for Loss function */
  Vec aux2;               /* Auxiliary state to eval loss */

  double loss_scaling_factor; /* Scaling the loss function value by this umber. Default=1.0 */

  public: 
    double current_err;
    Data* data;       /* Stores the data */

  public: 
    Learning(std::vector<int>& nlevels, LindbladType lindbladtype_, std::vector<std::string>& UDEmodel_str,  std::vector<int>& ncarrierwaves, std::vector<std::string>& learninit_str, Data* data, std::default_random_engine rand_engine, bool quietmode, double loss_scaling_factor);
    ~Learning();

    void resetLoss(){ loss_integral = 0.0; };
    double getLoss() { return loss_integral; };

    /* Get total number of learnable parameters */
    int getNParams(){ return nparams; };
    int getNParamsHamiltonian(){ return learnparamsH.size();};
    int getNParamsLindblad(){ return learnparamsL.size(); };
    int getNParamsTransfer(){ int sum=0; for (int i=0; i<learnparamsT.size();i++) sum+= learnparamsT[i].size(); return sum; };
    int getNParamsTransfer(int iosc){ return learnparamsT[iosc].size(); };

    /* Initialize learnable parameters. */
    void initLearnParams(int nparams, std::vector<std::string> learninit_str, std::default_random_engine rand_engine);

    /* Applies Hamiltonian and Lindblad UDE terms to input state (u,v) */
    void applyUDESystemMats(Vec u, Vec v, Vec uout, Vec vout);
    /* Adjoint Hamiltonian and Lindblad gradient: Sets (uout,vout) = dFWD^T *(u,v) */
    void applyUDESystemMats_diff(Vec u, Vec v, Vec uout, Vec vout);
    /* Gradient wrt learnable parameters from Hamiltonian and Lindblad parts: Sets grad += alpha * (dRHS(u,v)/dgamma)^T *(ubar, vbar) */
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);

    /* Applies the control transfer function inside oscillator->evalControl() */
    void applyUDETransfer(int oscilID, int cwID, double* Blt1, double* Blt2);
    /* Derivative of above function, applies within oscillator->evalControl_diff*/
    void applyUDETransfer_diff(int oscilID, int cwID, const double Blt1, const double Blt2, double& Blt1bar, double& Blt2bar, double* grad, double x_is_control);



    /* Assemble and view the learned SystemMat operators. */
    void writeOperators(std::string datadir);

    /* Copy optimization variable x into learnable parameter storage */
    void setLearnParams(const Vec x);
    /* Copy learnable parameters from storage into optimization variable x */
    void getLearnParams(double* x);

    /* Add to loss. Note: pulse_num is the global pulse number . */
    void addToLoss(double time, Vec x, int pulse_num);
    void addToLoss_diff(double time, Vec xbar, Vec xprimal, int pulse_num, double Jbar_loss);
};

