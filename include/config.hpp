#include <map>
#include <string>
#include <cstring>
#include <petsc.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cctype>
#include <vector>
#include <sstream>

#pragma once

class MapParam : public std::map<std::string, std::string>
{
  MPI_Comm comm;
  int mpi_rank;
  bool quietmode;

public:
  std::stringstream* log;

  /* Constructor */
  MapParam();
  MapParam(MPI_Comm comm_, std::stringstream& logstream, bool quietmode=false);
  /* Destructor */
  ~MapParam();
  
  /* Parse the config file. Stores keys and values in the map. */
  void ReadFile(std::string filename);

  /* Return mpirank */
  int GetMpiRank() const;

  /* Get config options (double, int, string, bool) */
  double GetDoubleParam(std::string key, double default_val = -1., bool warnme=true) const;
  int GetIntParam(std::string key, int default_val = -1, bool warnme=true, bool exportme = true) const;
  std::string GetStrParam(std::string key, std::string default_val = "", bool exportme = true, bool warnme=true) const;
  bool GetBoolParam(std::string key, bool default_val = false, bool exportme = true, bool warnme=true) const;
  void GetVecDoubleParam(std::string key, std::vector<double> &fillme, double default_val = 1e20, bool exportme = true, bool warnme = true) const;
  void GetVecIntParam(std::string key, std::vector<int> &fillme, int default_val = -1, bool warnme=true) const;
  void GetVecStrParam(std::string key, std::vector<std::string> &fillme, std::string default_val = "none", bool exportme = true, bool warnme=true) const;
};


template <typename T>
void export_param(int mpi_rank, std::stringstream& log, std::string key, T value)
{
  if (mpi_rank == 0)
  {
    log << key << " = " << value << std::endl;
  }
};

