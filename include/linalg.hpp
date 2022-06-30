#include <stdio.h>
#include <iostream> 
#include <math.h>
#include <assert.h>
#include <petscmat.h>
#include <vector>
#include "defs.hpp"
#pragma once

/* Abstract base class for vector representation */
class Vector {
  protected:
    int size;

  public: 
    Vector();
    Vector(int size);
    virtual ~Vector();

    void assemble() {};   /* Assembly needed for the Petsc vectors, see below. */
};

/* Vector based on Petsc's MPI Vec */
class Vec_MPI : Vector {
  Vec petscvec;        /* petsc vector */
  int ilow, iupp;  /* indices of vector elements that belong to this processor (excluding iupper) */

  public:
    Vec_MPI();
    Vec_MPI(int size);
    ~Vec_MPI();

    /* Calls Petsc's assembly routines. Needed after setting values */
    void assemble();
};

/* Vector based on Bjorn's handcoded OpenMP parallelization */
class Vec_OpenMP : Vector {
  public:
    Vec_OpenMP();
    ~Vec_OpenMP();
};

/* Abstract base class for sparse Matrix representation */
class SparseMatrix {
	public:
    SparseMatrix();
    virtual ~SparseMatrix();

    // void mult(const Vector& xin, Vector& xout) = 0;
};

/* Sparse Matrix using Petsc's AIJ matrices and MPI parallelization */
class Mat_MPI: public SparseMatrix {
  public:
    Mat_MPI();
    ~Mat_MPI();
};

/* Sparse Matrix using Bjorn's handcoded matrices and OpenMP parallelization */
class Mat_OpenMP : public SparseMatrix {
  public:
    Mat_OpenMP(); 
    ~Mat_OpenMP(); 
};

