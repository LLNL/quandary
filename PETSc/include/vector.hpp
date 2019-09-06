#include <iostream>
#include <cstring>
#pragma once

/*
 * Implements a vector structure 
 */
class Vector {
  protected:
    int dim;
    double *data;

  public:
    /* Various constructors */
    Vector();
    Vector(int dim_);
    Vector(int dim_, double fill_);
    /* Destructor */
    virtual ~Vector();

    /* Access and modify data */
    double operator()(int i) const;
    double &operator()(int i);
    Vector &operator+=(const Vector &v);
    Vector &operator-=(const Vector &v);
    Vector &operator*=(double alpha);

    /* Weighted sum: y <- alpha*x+beta*y, y=this */
    Vector &AXPBY(double alpha, double beta, const Vector &x);
    void CopyData(const Vector &v);
    void Fill(const double fill_);

    /* Norms */
    double Norm2() const;
    double NormInf() const;

    /* Get-functions */
    int GetDim() const { return dim; };
    double *GetData() const { return data; };

    /* Output routines */
    virtual void dump() const;
    virtual void dump(std::ostream &output) const;
    virtual void dump(std::string filename) const;
};

std::ostream &operator<<(std::ostream &output, const Vector &vec);


/*
 * Implements a twodimensional vector aka matrix
 */
class MultiVector : public Vector{

  int dimx; 
  int dimy;

  public:
    MultiVector();
    MultiVector(int dimx_, int dimy_, double fill_);
    ~MultiVector();

    /* Access */
    double operator()(int i, int j) const;
    double &operator()(int i, int j);

    /* Output routines */
    virtual void dump() const;
    virtual void dump(std::ostream &output) const;
    virtual void dump(std::string filename) const;
};