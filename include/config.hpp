#include <map>
#include <string>
#include <cstring>
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cctype>
#include <bits/stdc++.h> 

#pragma once

class MapParam : public std::map<std::string, std::string>
{
  MPI_Comm comm;
  int mpi_rank;

public:
  /* Constructor */
  MapParam();
  MapParam(MPI_Comm comm_);
  /* Destructor */
  ~MapParam();
  
  /* Parse the config file. Stores keys and values in the map. */
  void ReadFile(std::string filename);

  /* Return mpirank */
  int GetMpiRank() const;

  /* Get config options (double, int, string, bool) */
  double GetDoubleParam(std::string key, double default_val = -1.) const;
  int GetIntParam(std::string key, int default_val = -1) const;
  std::string GetStrParam(std::string key, std::string default_val = "") const;
  bool GetBoolParam(std::string key, bool default_val = false) const;
  void GetVecDoubleParam(std::string key, std::vector<double> &fillme, double default_val = 1e20) const;
};

