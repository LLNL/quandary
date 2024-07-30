#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
#include <fstream>
#include "util.hpp"
#include "defs.hpp"
#include <assert.h>
#include <petscts.h>
#include <vector>
#pragma once

/* Abstract base data class */
class Data{
  protected:
    int dim;              // Dimension of full vectorized system: N^2 for Lindblad, N for Schroedinger, or -1 if not learning.

    double dt;             /* Sample rate of the data */
    std::vector<Vec> data; /* List of all data point (rho_data) at each data time-step */
    std::vector<std::string> data_name; /* Name of the data files */

	public:
    Data();
    Data(std::vector<std::string> data_name, int dim);
    virtual ~Data();

    /* Get number of data elements */
    int getNData(){ return data.size(); };

    /* Get data time-spacing */
    double getDt(){ return dt; };

    /* Get data trajectory element */
    Vec getData(int dataID){ return data[dataID]; };
};

class SyntheticQuandaryData : public Data {
  public:
  SyntheticQuandaryData (std::vector<std::string> data_name, double data_tstop, int dim);
  ~SyntheticQuandaryData ();

  double loadData(double tstop = 1e+14);
};