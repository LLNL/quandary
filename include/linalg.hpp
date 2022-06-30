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
    int dim;

  public: 
    Vector();
    Vector(int dim);
    virtual ~Vector();

    /* Print the vector to screen */
    virtual void view() =0;
    /* Reset the vector to zero */
    virtual void setZero()=0;  

    virtual void getDistribution(int* ilow_ptr, int* iupp_ptr);

    /* Insert values into the vector at given index locations */
    virtual void setValues(std::vector<int>& indices, std::vector<double>& vals)=0;
};

/* Vector based on Petsc's MPI Vec */
class Vec_MPI : public Vector {
  Vec petscvec;        /* petsc vector */
  int ilow, iupp;  /* indices of vector elements that belong to this processor (excluding iupper) */

  public:
    Vec_MPI();
    Vec_MPI(int size);
    ~Vec_MPI();

    /* Print the vector to screen */
    void view();
    /* Reset the vector to zero */
    void setZero();  

    void getDistribution(int* ilow_ptr, int* iupp_ptr);

    /* Insert values into the vector */
    void setValues(std::vector<int>& indices, std::vector<double>& vals);
};

/* Vector based on Bjorn's handcoded OpenMP parallelization */
class Vec_OpenMP : public Vector {
  public:
    Vec_OpenMP();
    ~Vec_OpenMP();

    void view();
    void setZero();  
};

/* Abstract base class for sparse Matrix representation */
class SparseMatrix {
  int dim;  // Matrix Dimension. Assuming a square matrix here! dim = N^2 (Lindblad) or N (Schroedinger)

	public:
    SparseMatrix();
    SparseMatrix(int dim);
    virtual ~SparseMatrix();

    /* Print the vector to screen */
    virtual void view() =0;

    /* Insert matrix values at given index locations */
    virtual void setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals)=0;

    /* Insert or add one value at location (row, col) */
    virtual void setValue(int row, int col, double vals, bool add=false) {};

    /* Return distribution. Default: return 0,dim */
    virtual void getDistribution(int* ilow_ptr, int* iupp_ptr);

    /* Assembly of the matrix. Default: Do nothing */
    virtual void assemble() {};
};

/* Sparse Matrix using Petsc's AIJ matrices and MPI parallelization */
class Mat_MPI: public SparseMatrix {
  Mat PetscMat;
  int ilow, iupp; 

  public:
    Mat_MPI();
    Mat_MPI(int dim, int preallocate_per_row=0);
    ~Mat_MPI();

    void view();

    void setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals);

    /* Insert or add one value at location (row, col). This call MUST be followed by an assembly! */
    void setValue(int row, int col, double val, bool add=false);

    void getDistribution(int* ilow_ptr, int* iupp_ptr);

    void assemble();
};

/* Sparse Matrix using Bjorn's handcoded matrices and OpenMP parallelization */
class Mat_OpenMP : public SparseMatrix {
  public:
    Mat_OpenMP(); 
    ~Mat_OpenMP(); 

    void view();
    void setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals);
};

