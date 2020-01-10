#include "optimizer.hpp"

using namespace Ipopt;

OptimProblem::OptimProblem() {
    primalbraidapp  = NULL;
    adjointbraidapp = NULL;
    targetgate = NULL;
}

OptimProblem::OptimProblem(myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, Gate* targetgate_){
    primalbraidapp  = primalbraidapp_;
    adjointbraidapp = adjointbraidapp_;
    targetgate = targetgate_;
}

OptimProblem::~OptimProblem() {}

double OptimProblem::compare(Gate* gate, const double* state) {
    /* TODO: Compare distance of gate and state */
    /* For now, dummy objective function */
    double obj_val = 200.0 * state[1];
    return obj_val;
}

bool OptimProblem::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style){

  // n - number of design variables 
  n = 0;
  Hamiltonian* hamil = primalbraidapp->hamiltonian;
  for (int ioscil = 0; ioscil < hamil->getNOscillators(); ioscil++) {
      n += 2 * hamil->getOscillator(ioscil)->getNParam(); // Re and Im params for the i-th oscillator
  }
  
  // m - number of constraints 
  m = 0;          
  // number of non-zeros in jacobian of constraint
  nnz_jac_g = 0;  
  // number of non-zeros in hessian. Dense storage -> full hessian, but symmetric -> store lower left triangle
  nnz_h_lag = (int) n*(n+1)/2; 
  // use C-style indexing (0-based)
  index_style = TNLP::C_STYLE; 

  return true;
}



bool OptimProblem::get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u){

  /* no lower bound or upper bounds. Set bounds very big. */
  double bound = 2e19;

  for (int i=0; i<n; i++) {
    x_l[i] = -bound;
    x_u[i] =  bound;
  }

  assert(m=0);
  for (int i=0; i<m; i++) {
      g_l[i] = -bound;
      g_u[i] =  bound;
  }

  return true;
}


bool OptimProblem::get_starting_point(Index n, bool init_x, Number* x, bool init_z, Number* z_L, Number* z_U, Index m, bool init_lambda, Number* lambda){
  /* Make sure that only x requires an initial guess, not the dual variable z or the multiplier lambda. */
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);


  double *paramRe, *paramIm;
  int nparam;
  int j = 0;
  /* Iterate over oscillators */
  Hamiltonian* hamil = primalbraidapp->hamiltonian;
  for (int ioscil = 0; ioscil < hamil->getNOscillators(); ioscil++) {
      /* Get number of parameters of oscillator i */
      nparam = hamil->getOscillator(ioscil)->getNParam();
      /* Get pointers to parameters of oscillator i */
      paramRe = hamil->getOscillator(ioscil)->getParamsRe();
      paramIm = hamil->getOscillator(ioscil)->getParamsIm();
      /* Set initial condition */
      for (int iparam=0; iparam<nparam; iparam++) {
          x[j] = paramRe[iparam]; j++;
          x[j] = paramIm[iparam]; j++;
      }
  }

  return true;
}



bool OptimProblem::eval_f(Index n, const Number* x, bool new_x, Number& obj_value){

  Hamiltonian* hamil = primalbraidapp->hamiltonian;
  int dim = hamil->getDim();

  /* TODO: if (new_x) ... */

  /* Pass design vector x to oscillator */
  int nparam;
  double *paramRe, *paramIm;
  int j = 0;
  /* Iterate over oscillators */
  for (int ioscil = 0; ioscil < hamil->getNOscillators(); ioscil++) {
      /* Get number of parameters of oscillator i */
      nparam = hamil->getOscillator(ioscil)->getNParam();
      /* Get pointers to parameters of oscillator i */
      paramRe = hamil->getOscillator(ioscil)->getParamsRe();
      paramIm = hamil->getOscillator(ioscil)->getParamsIm();
      /* Set parameters */
      for (int iparam=0; iparam<nparam; iparam++) {
          paramRe[iparam] = x[j]; j++;
          paramIm[iparam] = x[j]; j++;
      }
  }

  /* TODO: Iterate over initial condition */
  int iinit = 0;
  obj_value = 0.0;

  /* Run simulation */
  primalbraidapp->SetInitialCondition(iinit);
  primalbraidapp->Drive();

  /* Compute objective function value */
  const double *finalstate = primalbraidapp->getState(primalbraidapp->total_time); // this returns NULL for all but the last processors! 
  if (finalstate != NULL) {
    /* Compare to target gate */
    obj_value += compare(targetgate, finalstate);

  }

  return true;
}



bool OptimProblem::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f){

    /* pass x to braid, forward simulation */
    double objective;
    eval_f(n, x, true, objective);

    /* run backward simulation */
    int iinit = 0;
    adjointbraidapp->SetInitialCondition(iinit);
    adjointbraidapp->Drive();

    /* Pass reduced gradient to ipopt */
    const double* grad_ptr = adjointbraidapp->getReducedGradientPtr();
    for (int i=0; i<n; i++) {
        grad_f[i] = grad_ptr[i];
    }
    
    return true;
}


bool OptimProblem::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
    assert(m=0);
    /* No constraints. Nothing to be done. */
    return true;
}


bool OptimProblem::eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index *jCol, Number* values){
    assert(m=0);
    /* No constraints. Nothing to be done. */
    return true;
}


//  We are using l-bfgs. So no hessian needed.
// bool OptimProblem::eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, const Number* lambda, bool new_lambda, Index nele_hess, Index* iRow, Index* jCol, Number* values){
//     return false;
// }

void OptimProblem::finalize_solution(SolverReturn status, Index n, const Number* x, const Number* z_L, const Number* z_U, Index m, const Number* g, const Number* lambda, Number obj_value, const IpoptData* ip_data,IpoptCalculatedQuantities* ip_cq){
    /* This is called after Ipopt finished. */
    /* TODO: pass x to someone, write to file... */
}