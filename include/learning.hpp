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

/* Generalized Gellmann matrices, diagonally shifted such tha G_00 = 0, optionally only upper part */
class GellmannBasis {
  protected:
    int dim_rho;   /* Dimension of the Hilbertspace (N)*/
    int dim;       /* N (if Schroedinger solver) or N^2 (if Lindblad) */
    int nbasis;    /* Total number of basis matrices */
    LindbladType lindbladtype;  // decides whether or not to vectorize the system matrices
    bool upper_only; // Optional: only get the upper diagonal part (including the diagonal itself)

    std::vector<Mat> BasisMat_Re; /* All (purely) real basis matrices. Size = dim_rho = N */ 
    std::vector<Mat> BasisMat_Im; /* All (purely) imaginary basis matrices. Size = dim_rho = N */ 

    std::vector<Mat> SystemMats_A;  // System matrix when applying the operator in Schroedinger's equation
    std::vector<Mat> SystemMats_B;  // System matrix when applying the operator in Schroedinger's equation

    Vec aux;     // Auxiliary vector to perform matvecs on Re(x) or Im(x)

  public:
     GellmannBasis(int dim_rho_, bool upper_only_, LindbladType lindbladtype_);
     virtual ~GellmannBasis();

    int getNBasis(){return nbasis;};
    int getNBasis_Re(){return BasisMat_Re.size();};
    int getNBasis_Im(){return BasisMat_Im.size();};
    Mat getBasisMat_Re(int id) {return BasisMat_Re[id];};
    Mat getBasisMat_Im(int id) {return BasisMat_Im[id];};

    virtual void assembleSystemMats()=0;

    virtual void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im) = 0;
    virtual void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im) = 0;
    virtual void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar) = 0;
};

/* Hamiltonian paramterization via generalized Gellman matrices, multiplied by (-i) and shifted s.t. G_00=0 */
class HamiltonianBasis : public GellmannBasis {

  Mat Operator_Re;  /* All assembled real operators */
  Mat Operator_Im;  /* All assembled imaginary operators */

  public:
    HamiltonianBasis(int dim_rho_, LindbladType lindbladtype);
    ~HamiltonianBasis();
    
    void assembleSystemMats();

    void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im);
    void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im);
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);

    void assembleOperator(std::vector<double>& learnparamsH_A, std::vector<double>& learnparamsH_B);
    Mat getOperator_Re() {return Operator_Re;};
    Mat getOperator_Im() {return Operator_Im;};
};

class LindbladBasis: public GellmannBasis {

  public:
    LindbladBasis(int dim_rho_);
    ~LindbladBasis();

    void assembleSystemMats();

    void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im);
    void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im);
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);
};

class Learning {

  int dim;              // Dimension of full vectorized system: N^2 for Lindblad, N for Schroedinger, or -1 if not learning.
  int dim_rho;               // Dimension of Hilbertspace = N
  LindbladType lindbladtype; // Switch for Lindblad vs Schroedinger solver

  HamiltonianBasis* hamiltonian_basis;  // Basis matrices for Hamiltonian term
  LindbladBasis* lindblad_basis;     // Basis matrices for Lindblad term 
  std::vector<double> learnparamsH_Re; // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsH_Im; // Learnable parameters for Hamiltonian
  std::vector<double> learnparamsL_Re; // Learnable parameters for Lindblad
  std::vector<double> learnparamsL_Im; // Learnable parameters for Lindblad
  
  int nparams;           /* Total Number of learnable paramters*/

  double data_dtAWG;     /* Sample rate of AWG data (default 4ns) */
  int data_ntime;        /* Number of data points in time */
  int loss_every_k;      /* Add to loss at every k-th timestep */
  std::vector<Vec> data; /* List of all data point (rho_data) at each data_dtAWG */

  Vec aux2;    // Auxiliary vector to perform matvecs on state x during Loss

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

    /* Initialize learnable parameters. */
    void initLearnableParams(std::vector<std::string> learninit_str, std::default_random_engine rand_engine);

    /* Applies UDE terms to input state (u,v) */
    void applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout);
    /* Adjoint gradient: Sets (uout,vout) = dFWD^T *(u,v) */
    void applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout);

    /* Reduced gradient for Hamiltonian part: 
       Sets grad += alpha * (dRHS(u,v)/dgamma)^T *(ubar, vbar) */
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar);

    /* Load data from file */
    void loadData(std::string data_name, double data_dtAWG, int data_ntime);

    /* Get data trajectory element */
    Vec getData(int id) {assert(data.size()>id); return data[id];};

    /* Get number of data elements */
    int getNData(){ return data.size(); };

    /* Assemble the learned operator. Allocates the (dense!) matrices Re(H) and Im(H), which hence must be destroyed after usage. */
    void viewOperators();

    /* Pass learnable parameters to storage learnparamsH_A and learnparamsH_B*/
    void setLearnParams(const Vec x);

    /* Copy learnable parameters from storage into x */
    void getLearnParams(double* x);

    /* Add to loss */
    void addToLoss(int timestepID, Vec x);
    void addToLoss_diff(int timestepID, Vec xbar, Vec xprimal, double Jbar_loss);
};

