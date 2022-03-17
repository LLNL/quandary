#include "defs.hpp"
#include <math.h>
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#include <bspline.hpp>
#ifdef WITH_PYTHON
#define PY_SSIZE_T_CLEAN
#include "Python/Python.h"
#endif
#pragma once

class PythonInterface{

  protected:

#ifdef WITH_PYTHON
    PyObject *pModule;  // The python module from which the Hamiltonian will be read
#endif
    LindbladType lindbladtype;            // Storing whether Lindblas solver or Schroedinger solver
    std::vector<int> ncontrolterms_store; // Storing the number of control terms per oscillator
    int dim_rho;                          // Dimension of the Hilbertspace. N!

	public:
    PythonInterface();
    PythonInterface(std::string python_file_, LindbladType lindbladtype_, int dim_rho_);
    ~PythonInterface();

  /* Receive the constant system Hamiltonian from "getHd" */
  // The python function "getHd" MUST return a LIST of floats!
  // Hd must be REAL valued!
  void receiveHd(Mat& Bd);

  /* Receive the time-varying system Hamiltonian part from "getHdt_real" and "getHdt_imag" */
  // These are Q*(Q+1)/2 Hamiltonians that will be applied as 
  //   J_kl cos(eta_kl t) * Hdt_real + i J_kl sin(eta_kl t) * Hdt_imag
  // TODO: Generalize for transfer functions u(t) and v(t) (receiveTransferHd, TODO.) 
  // The python function MUST return a LIST (length Q*(Q-1)/2) of LISTS (length dim^2) of floats!
  void receiveHdt(int noscillators, std::vector<Mat>& Ad_vec, std::vector<Mat>& Bd_vec);

  /* Receive control terms from "getHc_real" and "getHc_imag" */
  /* Fills up Ac_vec and Bc_vec, and return the number of control terms per oscillator in ncontrolterms */ 
  // getHc_re/im() MUST return a python list of lists of lists of float elements:
  //   for each oscillator k=0...Q-1: 
  //       for each control term i=0...C^k-1: 
  //           a list containing the flattened Hamiltonian Hc^k_i
  void receiveHc(int noscillators, Mat** Ac_vec, Mat** Bc_vec, std::vector<int>& ncontrolterms);

  /* Receive transfer functions from "getTransfer_real" and "getTransfer_imag" */
  // getTransfer() MUST return a python list of lists of [splines knots, and coeffs, and order]:
  //   for each oscillator k=0...Q-1: 
  //       for each control term i=0...C^k-1: 
  //            one transfer function u^k_i(x) (given in terms of spline knots (list), coefficients(list) and order (int)
  void receiveTransferHc(int noscillators,std::vector<std::vector<TransferFunction*>>& transfer_func_re, std::vector<std::vector<TransferFunction*>>& transfer_func_im);

};
