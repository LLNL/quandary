#pragma once

/* Available lindblad types */
enum LindbladType {NONE, DECAY, DEPHASE, BOTH};

/* Available types of initial conditions */
enum InitialConditionType {FROMFILE, PURE, DIAGONAL, BASIS};

/* Typye of objective functions */
enum ObjectiveType {GATE,             // Compare final state to linear gate transformation of initial cond.
                    EXPECTEDENERGY,   // Minimizes expected energy levels.
                    EXPECTEDENERGYb,   // Minimizes expected energy levels.
                    EXPECTEDENERGYc,   // Minimizes expected energy levels.
                    GROUNDSTATE};     // Compares final state to groundstate (full matrix)

/* Type of control fucntion evaluation: Rotating frame Real p(t), rotating frame imaginary q(t), or Lab frame f(t) */
enum ControlType {RE, IM, LAB};   


/* Linear solver */
enum LinearSolverType{
  GMRES,   // uses Petsc's GMRES solver
  NEUMANN   // uses Neuman power iterations 
};

/* Solver run type */
enum RunType {
  primal,            // Runs one objective function evaluation (forward)
  adjoint,           // Runs one gradient computation (forward & backward)
  optimization,      // Run optimization 
  none               // Don't run anything.
};

