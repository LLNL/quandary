#include "defs.hpp"
#include <math.h>
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#include <controlbasis.hpp>
#ifdef WITH_PYTHON
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif

#if PY_MAJOR_VERSION >= 3
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyString_FromString          PyUnicode_FromString
#endif

#pragma once

class PythonInterface{

  protected:

#ifdef WITH_PYTHON
    PyObject *pModule;  // The python module from which the Hamiltonian will be read
#endif
    LindbladType lindbladtype;            // Storing whether Lindblas solver or Schroedinger solver
    int dim_rho;                          // Dimension of the Hilbertspace. N!
    std::vector<int>ncontrol_real;
    std::vector<int>ncontrol_imag;
    int mpirank_world;   // Rank of global communicator

	public:
    PythonInterface();
    PythonInterface(std::string python_file_, LindbladType lindbladtype_, int dim_rho_);
    ~PythonInterface();

  /* Receive the constant system Hamiltonian from "getHd" */
  // The python function "getHd" MUST return a LIST of floats!
  // Hd must be REAL valued!
  void receiveHd(Mat& Bd);

  /* Receive the time-varying system Hamiltonian part from "getHdt_real" and "getHdt_imag" */
  // These are <L> Hamiltonians that will be applied as 
  //   u(t) * Hdt_real + i v(t) * Hdt_imag
  // for transfer functions u(t) and v(t) from receiveHdtTransfer 
  // The python function MUST return a LIST of LISTS of floats!
  void receiveHdt(std::vector<Mat>& Ad_vec, std::vector<Mat>& Bd_vec);

  /* Receive transfer functions from "getHdtTransfer_real" and "getHdtTransfer_imag" */
  // getTransfer() MUST return a python list of lists of [splines knots, and coeffs, and order]:
  //   for each Hdt term : 
  //            one transfer function u^k_i(x) (given in terms of spline knots (list), coefficients(list) and order (int)
  void receiveHdtTransfer(int nterms, std::vector<TransferFunction*>& transfer_Hdt_re, std::vector<TransferFunction*>& transfer_Hdt_im);

  /* Receive control terms from "getHc_real" and "getHc_imag" */
  // getHc_re/im() MUST return a python list of lists of lists of float elements:
  //   for each oscillator k=0...Q-1: 
  //       for each control term i=0...C^k-1: 
  //           a list containing the flattened Hamiltonian Hc^k_i
  void receiveHc(int noscillators, std::vector<std::vector<Mat>>& Ac_vec, std::vector<std::vector<Mat>>& Bc_vec);

  /* Receive transfer functions from "getHcTransfer_real" and "getHcTransfer_imag" */
  // getTransfer() MUST return a python list of lists of [splines knots, and coeffs, and order]:
  //   for each oscillator k=0...Q-1: 
  //       for each control term i=0...C^k-1: 
  //            one transfer function u^k_i(x) (given in terms of spline knots (list), coefficients(list) and order (int)
  void receiveHcTransfer(int noscillators,std::vector<std::vector<TransferFunction*>>& transfer_Hc_re,std::vector<std::vector<TransferFunction*>>& transfer_Hc_im);
};
