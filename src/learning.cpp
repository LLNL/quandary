#include "learning.hpp"


Learning::Learning(const int dim_){
    dim = dim_;

    /* Set an initial guess for the learnable parameters */
    // learnable Hamiltonian initialization
    int nparams_H = dim-1;  // N^2-1 many
    for (int i=0; i<nparams_H; i++){
        double val = 0.0;                // TODO
        learn_params_H.push_back(val);
    }
    // learnable Collapse operators
    int nparams_C = dim*(dim-1)/2;  // N^2(N^2-1)/2  many
    for (int i=0; i<nparams_C; i++){
        double val = 0.0;                // TODO
        learn_params_C.push_back(val);
    }

}

Learning::~Learning(){
    learn_params_H.clear();
    learn_params_C.clear();
}