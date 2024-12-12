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

// Switch to use the double-sum for Lindblad operators, vs only single sum
#define DOUBLESUM 1

class UDEmodel {
  protected:
    int dim_rho;   /* Dimension of the Hilbertspace (N)*/
    int dim;       /* N (if Schroedinger solver) or N^2 (if Lindblad) */
    int nparams;   /* Total number of learnable parameters */

    std::vector<Mat> SystemMats_A;  // System matrix for applying the Hamiltonian in the master equation
    std::vector<Mat> SystemMats_B;  // System matrix for applying the Hamiltonian in the master equation
    std::vector<Mat> Operator_Re;     // Learned operators
    std::vector<Mat> Operator_Im;     // Learned operators
    Vec aux;     // Auxiliary vector to perform matvecs on Re(x) or Im(x)

  public:
    UDEmodel();
    UDEmodel(int dim_rho_, LindbladType lindblad_type);
    virtual ~UDEmodel()=0;

    int getNParams(){return nparams;};

    virtual void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams) = 0;
    virtual void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams) = 0;
    virtual void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsH, int grad_skip) = 0;
    virtual void writeOperator(std::vector<double>& learnparams, std::string datadir) = 0;
};


/* Hamiltonian paramterization via generalized Gellman matrices */
class HamiltonianModel : public UDEmodel {
  bool shifted_diag; // Optional: Turn of shifting of the diagonal elements

  public:
    HamiltonianModel(int dim_rho_, bool shifted_diag_, LindbladType lindbladtype_);
    ~HamiltonianModel();
 
    void createSystemMats(LindbladType lindblad_type);

    void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams);
    void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams);
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsH, int grad_skip);
    void writeOperator(std::vector<double>& learnparams, std::string datadir);
};

class LindbladModel: public UDEmodel {
  int nbasis;   /* Number of basis operators */
  bool real_only;

  public:
    LindbladModel(int dim_rho_, bool shifted_diag_, bool upper_only_, bool real_only_);
    ~LindbladModel();

    int createSystemMats(bool upper_only, bool real_only, bool shifted_diag); // returns the total number of basis mats

    /* Assembles the system matrix operator */
    void evalOperator(std::vector<double>& learnparams);

    /* Index mapping for storing the lower triangular learnable matrix in a linear vector (vectorized column-wise) */
    inline int mapID(int i, int j){return i*nbasis - i*(i+1)/2 + j;}
    
    /* Evaluate the double-sum coefficient Gamma_ij */
    void getCoeffIJ(int i, int j, std::vector<double>& learnparams, double* aij_re, double* aij_im);

    void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams);
    void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams);
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL, int grad_skip);
    void writeOperator(std::vector<double>& learnparamsL, std::string datadir);
};
