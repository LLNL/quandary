#include <map>
#include <string>
#include <cstring>
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cctype>

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
  
  void ReadFile(std::string filename);
  double GetdoubleParam(std::string key, double default_val = -1.) const;
  int GetIntParam(std::string key, int default_val = -1) const;
  std::string GetStrParam(std::string key, std::string default_val = "") const;
  bool GetBoolParam(std::string key, bool default_val = false) const;
  int GetMpiRank() const;
};

