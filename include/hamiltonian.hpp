#include "mastereq.hpp"
#pragma once

class Hamiltonian{

protected:
   MasterEq* master_eq;
public:
   Hamiltonian( MasterEq* meq );
   virtual void apply( double t, Vector* u, Vector* v, Vector* uout, Vector* vout )=0;

};

class HamiltonianA : public Hamiltonian{
public:
   HamiltonianA( MasterEq* meq );
   virtual void apply( double t, Vector* u, Vector* v, Vector* uout, Vector* vout );
};

class HamiltonianB : public Hamiltonian{
   std::vector<double> m_J;
   std::vector<double> m_eta;
   int m_nsys, m_Ntot;
   double* m_p, *m_q, *m_csfact;
   Mat_OpenMP* m_smat, *m_kmat, *m_Jmatreal, *m_Jmatimag;
   double* m_hdiag;
public:
   HamiltonianB( MasterEq* meq );
   void setup_matrices();
   virtual void apply( double t, Vector* u, Vector* v, Vector* uout, Vector* vout );
};



