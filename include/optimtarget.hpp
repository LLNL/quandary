#include "defs.hpp"
#include "gate.hpp"
#pragma once

/* Collects stuff specifying the optimization target */
class OptimTarget{

    int dim;                       /* State dimension: either N^2 (if Lindblad solver), or N (if Schroedinger solver) */
    TargetType target_type;        /* Type of the optimization (pure-state preparation, or gate optimization) */
    ObjectiveType objective_type;  /* Type of the bjective function */
    Gate *targetgate;              /* The target gate (if any) */
    double purity_rho0;            /* Holds the purity of the initial state Tr(rho(0)^2) */
    int purestateID;               /* For pure-state preparation, this is the integer m for preparing e_m e_m^\dagger */
    std::string target_filename;   /* If a target is read from file, this holds it's filename. */
    Vec targetstate;   	           /* Holds the target state (unless its a pure one in which case this is NULL). 
                                      If target is a gate, this holds the transformed state VrhoV^\dagger.
                                      If target is read from file, this holds the target density matrix from that file. */

    Vec aux;      /* auxiliary vector needed when computing the objective for gate optimization */
    LindbladType lindbladtype;

  public:

    OptimTarget(int dim, int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_, std::string target_filename_, LindbladType lindbladtype_);
    ~OptimTarget();

    /* Get information on the type of optimization target */
    TargetType getTargetType(){ return target_type; };
    ObjectiveType getObjectiveType(){ return objective_type; };

    /* If gate optimization, this routine prepares the rotated target state VrhoV for a given initial state rho. Further, it stores the purity of rho(0) because it will be used to scale the Hilbertschmidt overlap for the JTrace objective function. */
    void prepare(const Vec rho);

    /* Evaluate the objective J. Note that J depends on the target state, which should be stored and ready before calling evalJ. The target state can be set with the 'prepare' routine. */
    /* Output is J_re and J_im. Generally imaginary part will be zero, except for the case of Schroedinger solver with Jtrace. */
    void evalJ(const Vec state, double* J_re_ptr, double* J_im_ptr);

    /* Derivative of evalJ. This updates the adjoint initial condition statebar */
    void evalJ_diff(const Vec state, Vec statebar, const double J_re_bar, const double J_im_bar);

    /* Finalze the objective function */
    double finalizeJ(const double obj_cost_re, const double obj_cost_im); 
    void finalizeJ_diff(const double obj_cost_re, const double obj_cost_im, double* obj_cost_re_bar, double* obj_cost_im_bar); 

    /* Frobenius distance F = 1/2 || targetstate - state ||^2_F */
    double FrobeniusDistance(const Vec state);
    void FrobeniusDistance_diff(const Vec state, Vec statebar, const double Jbar);

    /* Hilber-Schmidt overlap Tr(state * target^dagger), potentially scaled by purity of targetstate */
    /* Return real and imaginary parts in HS_re_ptr, Hs_im_ptr */
    void HilbertSchmidtOverlap(const Vec state, const bool scalebypurity, double* HS_re_ptr, double* Hs_im_ptr );
    void HilbertSchmidtOverlap_diff(const Vec state, Vec statebar, bool scalebypurity, const double HS_re_bar, const double HS_im_bar);
};

