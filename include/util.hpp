#include <petscmat.h>
#include <iostream>
#include <map>
#include <vector>
#ifdef WITH_SLEPC
#include <slepceps.h>
#endif

#pragma once


/**
 * @brief Sigmoid function for smooth transitions.
 *
 * @param width Transition width parameter
 * @param x Input value
 * @return double Sigmoid function value
 */
double sigmoid(double width, double x);

/**
 * @brief Derivative of sigmoid function.
 *
 * @param width Transition width parameter
 * @param x Input value
 * @return double Derivative of sigmoid function
 */
double sigmoid_diff(double width, double x);

/**
 * @brief Computes ramping factor for control pulse shaping.
 *
 * Computes smooth ramping factor for interval [tstart, tstop] using sigmoid
 * transitions with specified width tramp.
 *
 * @param time Current time
 * @param tstart Start time of interval
 * @param tstop Stop time of interval
 * @param tramp Ramping transition width
 * @return double Ramping factor between 0 and 1
 */
double getRampFactor(const double time, const double tstart, const double tstop, const double tramp);

/**
 * @brief Derivative of ramping factor with respect to stop time.
 *
 * @param time Current time
 * @param tstart Start time of interval
 * @param tstop Stop time of interval
 * @param tramp Ramping transition width
 * @return double Derivative with respect to tstop
 */
double getRampFactor_diff(const double time, const double tstart, const double tstop, const double tramp);

/**
 * @brief Returns vectorized index for matrix element (row,col).
 *
 * @param row Matrix row index
 * @param col Matrix column index
 * @param dim Matrix dimension
 * @return int Vectorized index for element (row,col)
 */
PetscInt getVecID(const PetscInt row, const PetscInt col, const PetscInt dim);

/**
 * @brief Maps index from essential level system to full-dimension system.
 *
 * @param i Index in essential level system
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Corresponding index in full-dimension system
 */
PetscInt mapEssToFull(const PetscInt i, const std::vector<size_t> &nlevels, const std::vector<size_t> &nessential);

/**
 * @brief Maps index from full dimension to essential dimension system.
 *
 * @param i Index in full dimension system
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Corresponding index in essential dimension system
 */
PetscInt mapFullToEss(const PetscInt i, const std::vector<size_t> &nlevels, const std::vector<size_t> &nessential);

/**
 * @brief Tests if density matrix index corresponds to an essential level.
 *
 * @param i Row/column index of density matrix
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Non-zero if index corresponds to essential level
 */
int isEssential(const int i, const std::vector<size_t> &nlevels, const std::vector<size_t> &nessential);

/**
 * @brief Tests if density matrix index corresponds to a guard level.
 *
 * A guard level is the highest energy level of an oscillator, used for
 * leakage detection and prevention.
 *
 * @param i Row/column index of density matrix
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Non-zero if index corresponds to guard level
 */
int isGuardLevel(const int i, const std::vector<size_t> &nlevels, const std::vector<size_t> &nessential);

/**
 * @brief Computes Kronecker product \f$Id \otimes A\f$.
 *
 * Computes the Kronecker product of an identity matrix with matrix A.
 * Output matrix must be pre-allocated with sufficient non-zeros A * dimI.
 *
 * @param[in] A Input matrix
 * @param[in] dimI Dimension of identity matrix
 * @param[in] alpha Scaling factor
 * @param[out] Out Output matrix \f$(Id \otimes A)\f$
 * @param[in] insert_mode INSERT_VALUES or ADD_VALUES
 * @return PetscErrorCode Error code
 */
PetscErrorCode Ikron(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode);

/**
 * @brief Computes Kronecker product \f$A \otimes Id\f$.
 *
 * Computes the Kronecker product of matrix A with an identity matrix.
 * Output matrix must be pre-allocated with sufficient non-zeros A * dimI.
 *
 * @param[in] A Input matrix
 * @param[in] dimI Dimension of identity matrix
 * @param[in] alpha Scaling factor
 * @param[out] Out Output matrix \f$(A \otimes Id)\f$
 * @param[in] insert_mode INSERT_VALUES or ADD_VALUES
 * @return PetscErrorCode Error code
 */
PetscErrorCode kronI(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode);

/**
 * @brief Computes general Kronecker product \f$A \otimes B\f$.
 *
 * Computes the Kronecker product of two arbitrary matrices A and B.
 * Works in PETSc serial mode only. Output matrix must be pre-allocated
 * and should be assembled afterwards.
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param alpha Scaling factor
 * @param Out Output matrix \f$(A \otimes B)\f$
 * @param insert_mode INSERT_VALUES or ADD_VALUES
 * @return PetscErrorCode Error code
 */
PetscErrorCode AkronB(const Mat A, const Mat B, const double alpha, Mat *Out, InsertMode insert_mode);

/**
 * @brief Tests if matrix A is anti-symmetric (A^T = -A).
 *
 * @param A Input matrix to test
 * @param tol Tolerance for comparison
 * @param flag Output flag indicating anti-symmetry
 * @return PetscErrorCode Error code
 */
PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag);

/**
 * @brief Tests if vectorized state represents a Hermitian matrix.
 *
 * For vectorized state x=[u,v] to represent a Hermitian matrix,
 * u must be symmetric and v must be anti-symmetric.
 *
 * @param x Vectorized state vector
 * @param tol Tolerance for comparison
 * @param flag Output flag indicating Hermiticity
 * @return PetscErrorCode Error code
 */
PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag);

/**
 * @brief Tests if vectorized state vector x=[u,v] represents matrix with trace 1.
 *
 * @param x Vectorized state vector
 * @param tol Tolerance for comparison
 * @param flag Output flag indicating unit trace
 * @return PetscErrorCode Error code
 */
PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag);

/**
 * @brief Performs all sanity tests on state vector.
 *
 * @param x State vector to test
 * @param time Current time for diagnostic output
 * @return PetscErrorCode Error code
 */
PetscErrorCode SanityTests(Vec x, PetscReal time);

/**
 * @brief Reads data vector from file.
 *
 * @param filename Name of file to read
 * @param var Array to store data
 * @param dim Dimension of data to read
 * @param quietmode Flag for reduced output
 * @param skiplines Number of header lines to skip
 * @param testheader Expected header string for validation
 * @return int Error code
 */
int read_vector(const char *filename, double *var, int dim, bool quietmode=false, int skiplines=0, const std::string testheader="");

/**
 * @brief Computes eigenvalues and eigenvectors of matrix A.
 *
 * Requires compilation with SLEPc for eigenvalue computations.
 *
 * @param A Input matrix
 * @param neigvals Number of eigenvalues to compute
 * @param eigvals Vector to store eigenvalues
 * @param eigvecs Vector to store eigenvectors
 * @return int Error code
 */
int getEigvals(const Mat A, const int neigvals, std::vector<double>& eigvals, std::vector<Vec>& eigvecs);

/**
 * @brief Tests if complex matrix A+iB is unitary.
 *
 * Tests whether (A+iB)(A+iB)^dagger = I for real matrices A and B.
 *
 * @param A Real part of complex matrix
 * @param B Imaginary part of complex matrix
 * @return bool True if matrix is unitary
 */
bool isUnitary(const Mat A, const Mat B);

/**
 * @brief Extends vector by repeating the last element.
 *
 * Template function that fills a vector to the specified size by
 * repeating the last element.
 *
 * @param fillme Vector to extend
 * @param tosize Target size for the vector
 */
template <typename Tval>
void copyLast(std::vector<Tval>& fillme, int tosize){
    // int norg = fillme.size();

    for (int i=fillme.size(); i<tosize; i++) 
      fillme.push_back(fillme[fillme.size()-1]);

    // if (norg < tosize) {
      // std::cout<< "I filled this: ";
      // for (int i=0; i<fillme.size(); i++) std::cout<< " " << fillme[i];
      // std::cout<<std::endl;
    // }
};

/**
 * @brief Prints error message to rank 0 cerr stream.
 *
 * @param mpi_rank MPI rank of the current process.
 * @param message Error message to log.
 */
void logErrorToRank0(int mpi_rank, const std::string& message);

/**
 * @brief Logs error message to rank 0 and exits program.
 *
 * Combines logErrorToRank0 and exit(1) into a single function for fatal errors
 * in MPI programs. Ensures only rank 0 outputs the error before terminating.
 *
 * @param mpi_rank Current MPI rank
 * @param message Error message to log before exiting
 */
void exitWithError(int mpi_rank, const std::string& message);

/**
 * @brief Prints output message to rank 0 cout stream.
 *
 * @param mpi_rank MPI rank of the current process.
 * @param message Error message to log.
 * @param quietmode Flag to suppress output when true.
 */
void logOutputToRank0(int mpi_rank, const std::string& message, bool quietmode = false);

/**
 * @brief Prints output message to rank 0 for given stream.
 *
 * @param mpi_rank MPI rank of the current process.
 * @param stream Stream to log output to.
 * @param message Error message to log.
 * @param quietmode Flag to suppress output when true.
 */
void logOutputToRank0(int mpi_rank, std::stringstream& stream, const std::string& message, bool quietmode = false);

/**
 * @brief Returns a lowercase version of the input string.
 *
 * @param str String to convert to lowercase.
 * @return std::string Lowercase string
 */
std::string toLower(std::string str);

/**
 * @brief Checks if string ends with specified suffix.
 *
 * @param str Input string to check.
 * @param suffix Suffix to look for.
 * @return bool True if string ends with suffix, false otherwise.
 */
bool hasSuffix(const std::string& str, const std::string& suffix);

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
      logOutputToRank0(mpi_rank, *default_stream, message, quiet_mode);
    } else {
      logOutputToRank0(mpi_rank, message, quiet_mode);
    }
  }

  void log(std::stringstream& stream, const std::string& message) const {
    logOutputToRank0(mpi_rank, stream, message, quiet_mode);
  }

  void logToConsole(const std::string& message) const {
    logOutputToRank0(mpi_rank, message, quiet_mode);
  }

  void error(const std::string& message) const {
    logErrorToRank0(mpi_rank, message);
  }

  void exitWithError(const std::string& message) const {
    ::exitWithError(mpi_rank, message);
  }

  bool isQuiet() const { return quiet_mode; }
  int getRank() const { return mpi_rank; }
};

/**
 * @brief Generic enum parsing utility with case-insensitive lookup.
 *
 * @param str String value to parse into enum
 * @param enum_map Map from string to enum values
 * @return std::optional<T> Parsed enum value or nullopt if not found
 */
template<typename T>
std::optional<T> parseEnum(const std::string& str, const std::map<std::string, T>& enum_map) {
  auto it = enum_map.find(toLower(str));
  if (it != enum_map.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}
