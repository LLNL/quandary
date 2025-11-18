#pragma once

#include <iostream>
#include <sstream>
#include <cstdlib>

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
  std::stringstream* default_stream;

public:
  MPILogger(int rank, bool quiet = false, std::stringstream* stream = nullptr)
    : mpi_rank(rank), quiet_mode(quiet), default_stream(stream) {}

  void log(const std::string& message) const {
    if (default_stream) {
      logToStream(*default_stream, message);
    } else {
      logToConsole(message);
    }
  }

  void log(std::stringstream& stream, const std::string& message) const {
    logToStream(stream, message);
  }

  void logToConsole(const std::string& message) const {
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

private:
  void logToStream(std::stringstream& stream, const std::string& message) const {
    if (!quiet_mode && mpi_rank == 0) {
      stream << message << std::endl;
    }
  }
};
