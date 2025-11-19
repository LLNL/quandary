#pragma once

#include <cstdlib>
#include <iostream>

/**
 * @brief MPI-aware logger that handles rank filtering and quiet mode.
 *
 * Encapsulates MPI rank and quiet mode to simplify logging calls throughout
 * the codebase. Only rank 0 outputs messages, and quiet mode can suppress output.
 */
class MPILogger {
 private:
  int mpi_rank;
  bool quiet_mode;

 public:
  MPILogger(int rank, bool quiet = false)
      : mpi_rank(rank), quiet_mode(quiet) {}

  void log(const std::string& message) const {
    if (!quiet_mode && mpi_rank == 0) {
      std::cout << message << std::endl;
    }
  }

  void error(const std::string& message) const {
    if (mpi_rank == 0) {
      std::cerr << "ERROR: " << message << std::endl;
    }
  }

  void exitWithError(const std::string& message) const {
    error(message);
    exit(1);
  }

  bool isQuiet() const { return quiet_mode; }
  int getRank() const { return mpi_rank; }
};
