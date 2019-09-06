#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "vector.hpp"

using namespace std;

Vector::Vector(){
  dim = 0;
  data = NULL;
}


Vector::Vector(int dim_){
  dim = dim_;
  data = new double[dim];
  this->Fill(0.0);
}

Vector::Vector(int dim_, double fill_){
  dim = dim_;
  data = new double[dim];
  this->Fill(fill_);
}

Vector::~Vector(){
  /* Free data, if it has been allocated. */
  if (dim>0){
    delete [] data;
  }
}

void Vector::Fill(const double fill_){
  for (int i=0; i<dim; i++){
    data[i] = fill_;
  }
}

double Vector::operator()(int i) const
{
  return data[i];
}

double &Vector::operator()(int i)
{
  return data[i];
}

Vector &Vector::operator+=(const Vector &v)
{
  for (int i = 0; i < dim; i++)
    data[i] += v.data[i];
  return *this;
}

Vector &Vector::operator-=(const Vector &v)
{
  for (int i = 0; i < dim; i++)
    data[i] -= v.data[i];
  return *this;
}

Vector &Vector::operator*=(double alpha)
{
  for (int i = 0; i < dim; i++)
    data[i] *= alpha;
  return *this;
}

// y <- alpha*x+beta*y, y=this
Vector &Vector::AXPBY(double alpha, double beta, const Vector &x)
{
  for (int i = 0; i < dim; i++)
    data[i] = alpha * x.data[i] + beta * data[i];
  return *this;
}

void Vector::CopyData(const Vector &v)
{
  memcpy(data, v.GetData(), dim * sizeof(double));
}

double Vector::Norm2() const
{
  double res = 0;
  for (int i = 0; i < dim; i++)
    res += data[i] * data[i];
  return sqrt(res);
}

double Vector::NormInf() const
{
  double res = 0;
  for (int i = 0; i < dim; i++)
    res = max(res, abs(data[i]));
  return res;
}

void Vector::dump() const
{
  this->dump(cout);
}

void Vector::dump(ostream &output) const
{
  output << data[0];
  for (int i = 1; i < dim; i++)
    output << " " << data[i];
}

void Vector::dump(string filename) const
{
  ofstream file;
  file.open(filename.c_str());
  file << setprecision(20);
  this->dump(file);
  file << endl;
  file.close();
}

ostream &operator<<(ostream &output, const Vector &vec)
{
  vec.dump(output);
  return output;
}

MultiVector::MultiVector() : Vector() {
  dimx = 0;
  dimy = 0;
}

MultiVector::MultiVector(int dimx_, int dimy_, double fill_) : Vector(dimx_*dimy_, fill_) {
  dimx = dimx_;
  dimy = dimy_;
}

MultiVector::~MultiVector(){}



double MultiVector::operator()(int i, int j) const {
  return data[i*dimy + j];
}

double &MultiVector::operator()(int i, int j) {
  return data[i*dimy + j];
}