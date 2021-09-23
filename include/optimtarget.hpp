#include "defs.hpp"
#include "gate.hpp"
#pragma once

/* Collects stuff specifying the optimization target */
class OptimTarget{

    TargetType target_type;        /* Type of the optimization (pure-state preparation, or gate optimization) */
    ObjectiveType objective_type;  /* Type of the bjective function */
    Gate *targetgate;              /* The target gate (if any) */
    int purestateID;               /* For pure-state preparation, this is the integer m for preparing e_m e_m^\dagger */
    std::string target_filename;   /* If a target is read from file, this holds it's filename. */
    Vec targetstate;   	           /* Holds the target state (unless its a pure one in which case this is NULL). 
                                      If target is a gate, this holds the transformed state VrhoV^\dagger.
                                      If target is read from file, this holds the target density matrix from that file. */

    Vec aux;      /* auxiliary vector needed when computing the objective for gate optimization */

  public:

    OptimTarget(int dim, int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_, std::string target_filename_);
    ~OptimTarget();

    /* Get information on the type of optimization target */
    TargetType getType(){ return target_type; };

    /* If gate optimization, this routine prepares the rotated target state VrhoV for a given initial state rho */
    void prepare(const Vec rho);

    /* Evaluate the objective J */
    /* Note that J depends on the target state which itself can depend on the initial state. Therefor, the targetstate should be computed within 'prepare' routine! */
    double evalJ(const Vec state);

    /* Evaluate the fidelity Tr(rhotarget^\dagger rho) */
    double evalFidelity(const Vec state);

    /* Derivative of evalJ. This updates the adjoint initial condition statebar */
    void evalJ_diff(const Vec state, Vec statebar, const double Jbar);

    /* Frobenius distance F = 1/2 || targetstate - state ||^2_F */
    double FrobeniusDistance(const Vec state);
    void FrobeniusDistance_diff(const Vec state, Vec statebar, const double Jbar);

    /* Hilber-Schmidt overlap Tr(targetstate^\dagger * state), potentially scaled by purity of targetstate */
    double HilbertSchmidtOverlap(const Vec state, bool scalebypurity);
    void HilbertSchmidtOverlap_diff(const Vec state, Vec statebar, const double Jbar, bool scalebypurity);
};

