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

class GellmannBasis {
  protected:
    int dim_rho;   /* Dimension of the Hilbertspace (N)*/
    int dim;       /* N (if Schroedinger solver) or N^2 (if Lindblad) */
    int nbasis;    /* Total number of basis matrices */
    LindbladType lindbladtype;  // decides whether or not to vectorize the system matrices
    bool upper_only; // Optional: only get the upper diagonal part (including the diagonal itself)
    bool shifted_diag; // Optional: Turn of shifting of the diagonal elements

    std::vector<Mat> BasisMat_Re; /* All (purely) real basis matrices. Size = dim_rho = N */ 
    std::vector<Mat> BasisMat_Im; /* All (purely) imaginary basis matrices. Size = dim_rho = N */ 
    Mat Id;			  /* Identity matrix */

    std::vector<Mat> SystemMats_A;  // System matrix when applying the operator in Schroedinger's equation
    std::vector<Mat> SystemMats_B;  // System matrix when applying the operator in Schroedinger's equation

    int nparams;

    Vec aux;     // Auxiliary vector to perform matvecs on Re(x) or Im(x)

  public:
     GellmannBasis(int dim_rho_, bool upper_only_, bool shifted_diag_, LindbladType lindbladtype_);
     virtual ~GellmannBasis();

    int getNBasis(){return nbasis;};
    int getNBasis_Re(){return BasisMat_Re.size();};
    int getNBasis_Im(){return BasisMat_Im.size();};
    Mat getBasisMat_Re(int id) {return BasisMat_Re[id];};
    Mat getBasisMat_Im(int id) {return BasisMat_Im[id];};
    Mat getIdentity(){return Id;};
    int getNParams(){return nparams;};

    virtual void assembleSystemMats()=0;

    virtual void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im) = 0;
    virtual void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im) = 0;
    virtual void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL_Re, int skipID=0) = 0;

    virtual void printOperator(std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im, std::string datadir) = 0;
};

/* Hamiltonian paramterization via generalized Gellman matrices */
class HamiltonianBasis {
  int dim_rho;   /* Dimension of the Hilbertspace (N)*/
  int dim;       /* N (if Schroedinger solver) or N^2 (if Lindblad) */
  bool shifted_diag; // Optional: Turn of shifting of the diagonal elements

  int nparams;  /* Total number of learnable parameters */

  std::vector<Mat> SystemMats_A;  // System matrix for applying the Hamiltonian in the master equation
  std::vector<Mat> SystemMats_B;  // System matrix for applying the Hamiltonian in the master equation

  Mat Operator_Re;  
  Mat Operator_Im;  

  Vec aux;     // Auxiliary vector to perform matvecs on Re(x) or Im(x)

  public:
    HamiltonianBasis(int dim_rho_, bool shifted_diag_, LindbladType lindbladtype_);
    ~HamiltonianBasis();

    int getNParams(){return nparams;};
    // int getNBasis_Re(){return SystemMats_B.size();};
    // int getNBasis_Im(){return SystemMats_A.size();};
    
    void createSystemMats(LindbladType lindblad_type);

    void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams);
    void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams);
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsH);

    void printOperator(std::vector<double>& learnparams, std::string datadir);
};

class LindbladBasis: public GellmannBasis {
  Mat Operator;  

  public:
    LindbladBasis(int dim_rho_, bool shifted_diag_);
    ~LindbladBasis();

    void assembleSystemMats();

    void applySystem(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im);
    void applySystem_diff(Vec u, Vec v, Vec uout, Vec vout, std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im);
    void dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar, std::vector<double>& learnparamsL_Re, int skipID=0);

    std::vector<Mat> getSystemMats_A() {return SystemMats_A;};
    std::vector<Mat> getSystemMats_B() {return SystemMats_B;};

    void printOperator(std::vector<double>& learnparams_Re, std::vector<double>& learnparams_Im, std::string datadir);

    void evalOperator(std::vector<double>& learnparams_Re);
    int mapID(int i, int j){return i*BasisMat_Re.size() - i*(i+1)/2 + j;}
};
