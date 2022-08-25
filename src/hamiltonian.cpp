#include "hamiltonian.hpp"

Hamiltonian::Hamiltonian( MasterEq* meq )
{
   master_eq = meq;
}

HamiltonianA::HamiltonianA( MasterEq* meq ) : Hamiltonian(meq)
{
}

void HamiltonianA::apply( double t, Vector* u, Vector* v, Vector* uout, Vector* vout )
{
  // Constant part uout = Adu - Bdv
  master_eq->Bd_LA->mult(v, uout);
  uout->scale(-1.0);
  master_eq->Ad_LA->mult(u, uout, true);
  // Constant part vout = Adv + Bdu
  master_eq->Ad_LA->mult(v, vout);
  master_eq->Bd_LA->mult(u, vout, true);

  /* Control terms and Jaynes-Cummings coupling terms */
  int id_kl = 0; // index for accessing Ad_kl inside Ad_vec
  for (int iosc = 0; iosc < master_eq->nlevels.size(); iosc++) {

    /* Get controls */
    double p, q;
    if (t>=0) master_eq->getOscillator(iosc)->evalControl(t, &p, &q);
     else {
       p = 0.1;
       q = 0.1;
     }

    // uout += q^k*Acu
    master_eq->Ac_LA_vec[iosc]->mult(u, master_eq->aux_LA);
    uout->add(q, master_eq->aux_LA);
    // uout -= p^kBcv
    master_eq->Bc_LA_vec[iosc]->mult(v, master_eq->aux_LA);
    uout->add(-1.*p, master_eq->aux_LA);
    // vout += q^kAcv
    master_eq->Ac_LA_vec[iosc]->mult(v, master_eq->aux_LA);
    vout->add(q, master_eq->aux_LA);
    // vout += p^kBcu
    master_eq->Bc_LA_vec[iosc]->mult(u, master_eq->aux_LA);
    vout->add(p, master_eq->aux_LA);

    // Coupling terms
    for (int josc=iosc+1; josc<master_eq->nlevels.size(); josc++){
       double Jkli = master_eq->getJkl(id_kl); 
      if (fabs(Jkli) > 1e-12) {

        double etakl = master_eq->getEta(id_kl);
        double coskl = cos(etakl * t);
        double sinkl = sin(etakl * t);
        // uout += J_kl*sin*Adklu
        master_eq->Ad_LA_vec[id_kl]->mult(u, master_eq->aux_LA);
        uout->add(Jkli*sinkl, master_eq->aux_LA);
        // uout += -Jkl*cos*Bdklv
        master_eq->Bd_LA_vec[id_kl]->mult(v, master_eq->aux_LA);
        uout->add(-Jkli*coskl, master_eq->aux_LA);
        // vout += Jkl*cos*Bdklu
        master_eq->Bd_LA_vec[id_kl]->mult(u, master_eq->aux_LA);
        vout->add(Jkli*coskl, master_eq->aux_LA);
        //vout += Jkl*sin*Adklv
        master_eq->Ad_LA_vec[id_kl]->mult(v, master_eq->aux_LA);
        vout->add(Jkli*sinkl, master_eq->aux_LA);
      }
      id_kl++;
    }
  }

}


//-----------------------------------------------------------------------
HamiltonianB::HamiltonianB( MasterEq* meq ) : Hamiltonian(meq)
{
   m_nsys    = master_eq->getNOscillators();
   setup_matrices();
}

//-----------------------------------------------------------------------
void HamiltonianB::setup_matrices()
{
   m_smat = new Mat_OpenMP[m_nsys];
   m_kmat = new Mat_OpenMP[m_nsys];
   std::vector<int> N(m_nsys);
   m_Ntot = 1;
   int nmax = 0;
   for( int k=0; k < m_nsys; k++ )
   {
      N[k] = master_eq->getOscillator(k)->getNLevels();
      m_Ntot *= N[k];
      nmax = N[k]>nmax?N[k]:nmax;
   }
   std::vector<double> cof(nmax), cof4(nmax);
   for( int l=0; l < nmax; l++ )
   {
      cof[l]  = l;
      cof4[l] = l*(l-1);
   }
   std::vector<int> off(m_nsys);
   off[m_nsys-1]=1;
   for( int k=m_nsys-2; k>=0; k-- )
      off[k] = N[k+1]*off[k+1];

// 1. Setup diagonal matrix
   m_hdiag = new double[m_Ntot];
   int* i=new int[m_nsys];
   for( int ind=0 ; ind < m_Ntot ;ind++ )
   {
      int t0=ind;
      for( int k=0; k<m_nsys; k++ )
      {
         i[k]=t0/off[k];
         t0=t0-i[k]*off[k];
      }
      m_hdiag[ind] = 0;
      int km=0;
      for( int k=0 ; k<m_nsys; k++ )
      {
         double detuning = master_eq->getOscillator(k)->getDetuning();    
         double xi=master_eq->getOscillator(k)->getSelfkerr();    
         m_hdiag[ind] += detuning*cof[i[k]]-0.5*xi*cof4[i[k]];
         for( int m=k+1; m<m_nsys; m++ )
         {
            double ximix=master_eq->getCrosskerr(km);
            m_hdiag[ind] -= ximix*cof[i[k]]*cof[i[m]];
            km++;
         }
      }
   }
   delete[] i;
   
// 2. Setup matrices A+A^* and A-A^* 
   std::vector<double*> alpha(m_nsys);
   for( int k=0; k < m_nsys; k++ )
   {
      alpha[k] = new double[N[k]-1];
      for( int j=0; j < N[k]-1 ; j++ )
         alpha[k][j] = sqrt(j+1.0);
   }

   for( int k=0; k < m_nsys; k++ )
   {
      m_smat[k].setup( k, off[k], N, alpha[k], false );
      m_kmat[k].setup( k, off[k], N, alpha[k], true );
   }
   m_p = new double[m_nsys];
   m_q = new double[m_nsys];

  // 3. Setup Jaynes-Cummings coupling term
   m_J.clear();
   std::vector<int> kmix, mmix;
   int ind=0;
   for( int k=0 ; k < m_nsys; k++ )
      for( int m=k+1; m < m_nsys; m++ )
      {
         if( fabs(master_eq->getJkl(ind)) > 1e-12 )
         {
            m_J.push_back(master_eq->getJkl(ind)); 
            m_eta.push_back(master_eq->getEta(ind));
            kmix.push_back(k);
            mmix.push_back(m);
         }
         ind++;
      }
   if( m_J.size() > 0 )
   {
      m_Jmatreal = new Mat_OpenMP[m_J.size()];
      m_Jmatimag = new Mat_OpenMP[m_J.size()];
      for( int l=0; l < m_J.size(); l++ )
      {
         int k=kmix[l], m=mmix[l];
         m_Jmatimag[l].setupJ( N, k, alpha[k], off[k], m, alpha[m], off[m], false );
         m_Jmatreal[l].setupJ( N, k, alpha[k], off[k], m, alpha[m], off[m], true );
      }
      m_csfact = new double[m_J.size()];
   }
   for( int k=0; k < m_nsys; k++ )
   {
      delete[] alpha[k];
   }
}

//-----------------------------------------------------------------------
void HamiltonianB::apply( double t, Vector* u, Vector* v, Vector* uout, Vector* vout )
{
   double* up=u->c_ptr(), *vp=v->c_ptr(), *uoutp=uout->c_ptr(), *voutp=vout->c_ptr();
   int Jsize=m_J.size();
   for( int l=0 ; l<Jsize; l++ )
      m_csfact[l] = m_J[l]*cos(m_eta[l]*t);

   for( int iosc=0; iosc < m_nsys; iosc++ )
   if (t>=0) 
      master_eq->getOscillator(iosc)->evalControl(t, &m_p[iosc], &m_q[iosc]);
   else 
   {
      m_p[iosc] = 0.1;
      m_q[iosc] = 0.1;
   }

#pragma omp parallel for
   for( int i=0 ; i < m_Ntot; i++ )
   {
      double h1p = -m_hdiag[i]*vp[i];
      double h2p =  m_hdiag[i]*up[i];
      for( int k=0; k < m_nsys; k++)
      {
         for( int ind=m_kmat[k].m_rowstarts[i]; ind < m_kmat[k].m_rowstarts[i+1];ind++)
         {
            int j=m_kmat[k].m_cols[ind];
            h1p -= m_p[k]*m_kmat[k].m_elements[ind]*vp[j];
            h2p += m_p[k]*m_kmat[k].m_elements[ind]*up[j];
         }
      }
      for( int l=0; l < Jsize; l++ )
      {
         for( int ind=m_Jmatreal[l].m_rowstarts[i]; ind < m_Jmatreal[l].m_rowstarts[i+1];ind++)
         {
            int j=m_Jmatreal[l].m_cols[ind];
            h1p -= m_csfact[l]*m_Jmatreal[l].m_elements[ind]*vp[j];
            h2p += m_csfact[l]*m_Jmatreal[l].m_elements[ind]*up[j];
         }
      }
      uoutp[i] = h1p;
      voutp[i] = h2p;
   }

   for( int l=0 ; l<Jsize; l++ )
      m_csfact[l] = m_J[l]*sin(m_eta[l]*t);
#pragma omp parallel for
   for( int i=0 ; i < m_Ntot; i++ )
   {
      double h1p=0;
      double h2p=0;
      for( int k=0; k < m_nsys; k++)
      {
         for( int ind=m_smat[k].m_rowstarts[i]; ind < m_smat[k].m_rowstarts[i+1];ind++)
         {
            int j=m_smat[k].m_cols[ind];
            h1p += m_q[k]*m_smat[k].m_elements[ind]*up[j];
            h2p += m_q[k]*m_smat[k].m_elements[ind]*vp[j];
         }
      }
      for( int l=0; l < Jsize; l++ )
      {
         for( int ind=m_Jmatimag[l].m_rowstarts[i]; ind < m_Jmatimag[l].m_rowstarts[i+1];ind++)
         {
            int j=m_Jmatimag[l].m_cols[ind];
            h1p += m_csfact[l]*m_Jmatimag[l].m_elements[ind]*up[j];
            h2p += m_csfact[l]*m_Jmatimag[l].m_elements[ind]*vp[j];
         }
      }
      uoutp[i] += h1p;
      voutp[i] += h2p;
   }
   u->restore_ptr(up);
   v->restore_ptr(vp);
   uout->restore_ptr(uoutp);
   vout->restore_ptr(voutp);
}

