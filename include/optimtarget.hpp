#include "defs.hpp"
#include "gate.hpp"
#pragma once

/* Collects stuff specifying the optimization target */
class OptimTarget{

    TargetType target_type;        /* Type of the optimization (pure-state preparation, or gate optimization) */
    ObjectiveType objective_type;  /* Type of the bjective function */
    Gate *targetgate;              /* The target gate (if any) */
    int purestateID;               /* The integer m for pure-state preparation of the m-the state */

    Vec aux;     /* auxiliary vector needed when computing the objective for gate optimization */

  public:
    Vec VrhoV;   /* For gate optimization: holds the transformed state VrhoV^\dagger */

    OptimTarget(int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_);
    ~OptimTarget();

    /* Get information on the type of optimization target */
    TargetType getType(){ return target_type; };
    /* Get information on the type of objective function */
    ObjectiveType getObjective() { return objective_type; };
    /* Get the target gate */
    Gate* getGate(){ return targetgate; };
    /* Get the pure state target */
    int getPureID(){ return purestateID; };

    /* Frobenius distance F = 1/2 || VrhoV - state ||^2_F */
    double FrobeniusDistance(const Vec state);
    void FrobeniusDistance_diff(const Vec state, Vec statebar, const double Jbar);

    /* Hilber-Schmidt overlap Tr(VrhoV^\dagger * state), potentially scaled by purity of VrhoV */
    double HilbertSchmidtOverlap(const Vec state, bool scalebypurity);
    void HilbertSchmidtOverlap_diff(const Vec state, Vec statebar, const double Jbar, bool scalebypurity);
};

