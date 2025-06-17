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

/**
 * @brief Configuration parameter management class.
 *
 * The `Config` class extends std::map to provide functionality for reading
 * configuration files and retrieving parameters with type conversion and default values.
 * It supports MPI communication and logging capabilities.
 */
class Config : public std::map<std::string, std::string> {
  protected:

  MPI_Comm comm; ///< MPI communicator for parallel operations.
  int mpi_rank; ///< MPI rank of the current process.
  bool quietmode; ///< Flag to control verbose output.

  public:
  std::stringstream* log; ///< Pointer to log stream for output messages.

  Config();

  /**
   * @brief Constructor with MPI communicator and logging.
   *
   * @param comm_ MPI communicator for parallel operations.
   * @param logstream Reference to the log stream for output.
   * @param quietmode Flag to enable quiet mode (default: false).
   */
  Config(MPI_Comm comm_, std::stringstream& logstream, bool quietmode=false);

  ~Config();
  
  /**
   * @brief Parses a configuration file and stores key-value pairs in the map.
   *
   * @param filename Path to the configuration file to be parsed.
   */
  void ReadFile(std::string filename);

  /**
   * @brief Retrieves the MPI rank of the current process.
   *
   * @return int MPI rank.
   */
  int GetMpiRank() const;

  /**
   * @brief Retrieves a double parameter from the configuration.
   *
   * @param key Parameter name to look up.
   * @param default_val Default value if parameter is not found (default: -1.0).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   * @return double Parameter value or default value.
   */
  double GetDoubleParam(std::string key, double default_val = -1., bool warnme=true) const;
  /**
   * @brief Retrieves an integer parameter from the configuration.
   *
   * @param key Parameter name to look up.
   * @param default_val Default value if parameter is not found (default: -1).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   * @param exportme Flag to enable logging of the parameter value (default: true).
   * @return int Parameter value or default value.
   */
  int GetIntParam(std::string key, int default_val = -1, bool warnme=true, bool exportme = true) const;
  /**
   * @brief Retrieves a string parameter from the configuration.
   *
   * @param key Parameter name to look up.
   * @param default_val Default value if parameter is not found (default: "").
   * @param exportme Flag to enable logging of the parameter value (default: true).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   * @return std::string Parameter value or default value.
   */
  std::string GetStrParam(std::string key, std::string default_val = "", bool exportme = true, bool warnme=true) const;
  /**
   * @brief Retrieves a boolean parameter from the configuration.
   *
   * @param key Parameter name to look up.
   * @param default_val Default value if parameter is not found (default: false).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   * @return bool Parameter value or default value.
   */
  bool GetBoolParam(std::string key, bool default_val = false, bool warnme=true) const;
  /**
   * @brief Retrieves a vector of double parameters from the configuration.
   *
   * @param key Parameter name to look up.
   * @param fillme Reference to vector to be filled with parameter values.
   * @param default_val Default value for each element if parameter is not found (default: 1e20).
   * @param exportme Flag to enable logging of the parameter values (default: true).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   */
  void GetVecDoubleParam(std::string key, std::vector<double> &fillme, double default_val = 1e20, bool exportme = true, bool warnme = true) const;
  /**
   * @brief Retrieves a vector of integer parameters from the configuration.
   *
   * @param key Parameter name to look up.
   * @param fillme Reference to vector to be filled with parameter values.
   * @param default_val Default value for each element if parameter is not found (default: -1).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   */
  void GetVecIntParam(std::string key, std::vector<int> &fillme, int default_val = -1, bool warnme=true) const;
  /**
   * @brief Retrieves a vector of string parameters from the configuration.
   *
   * @param key Parameter name to look up.
   * @param fillme Reference to vector to be filled with parameter values.
   * @param default_val Default value for each element if parameter is not found (default: "none").
   * @param exportme Flag to enable logging of the parameter values (default: true).
   * @param warnme Flag to enable warning if parameter is not found (default: true).
   */
  void GetVecStrParam(std::string key, std::vector<std::string> &fillme, std::string default_val = "none", bool exportme = true, bool warnme=true) const;
};


/**
 * @brief Template function to export parameter values to the log stream.
 *
 * This function logs parameter key-value pairs to the provided log stream,
 * but only from MPI rank 0 to avoid duplicate output in parallel runs.
 *
 * @tparam T Type of the parameter value.
 * @param mpi_rank MPI rank of the current process.
 * @param log Reference to the log stream.
 * @param key Parameter name.
 * @param value Parameter value to be logged.
 */
template <typename T>
void export_param(int mpi_rank, std::stringstream& log, std::string key, T value)
{
  if (mpi_rank == 0)
  {
    log << key << " = " << value << std::endl;
  }
};

