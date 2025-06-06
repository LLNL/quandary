#include <stdio.h>
#include <iostream> 
#include <math.h>
#include <assert.h>
#include <petscmat.h>
#include <vector>
#include "defs.hpp"
#include "util.hpp"
#pragma once

/**
 * @brief Base class for quantum gate operations.
 *
 * The Gate class provides the fundamental interface for quantum gate operations
 * in quantum optimal control. It handles the transformation V \rho V^\dagger where V is
 * the unitary gate matrix and \rho is the quantum state density matrix.
 */
class Gate {
  protected:
    Mat V_re, V_im; ///< Input: Real and imaginary parts of V_target, non-vectorized, essential levels only.
    Vec rotA, rotB; ///< Input: Diagonal elements of real and imaginary rotational matrices.

    std::vector<int> nessential; ///< Number of essential levels per oscillator.
    std::vector<int> nlevels; ///< Total number of levels per oscillator.
    int mpirank_petsc; ///< MPI rank in PETSc communicator.
    int mpirank_world; ///< MPI rank in world communicator.

    bool quietmode; ///< Flag to suppress output messages.

    int dim_ess; ///< Dimension of target gate matrix (non-vectorized), essential levels only.
    int dim_rho; ///< Dimension of system matrix rho (non-vectorized), all levels, N.

    double final_time; ///< Final time T. Time of gate rotation.
    std::vector<double> gate_rot_freq; ///< Frequencies of gate rotation (rad/time). Often same as rotational frequencies.

    LindbladType lindbladtype; ///< Type of Lindblad operators for open system dynamics.


  private:
    Mat VxV_re, VxV_im;     ///< Real and imaginary parts of vectorized gate G=\bar V \kron V.
    Vec x;                  ///< Auxiliary vector for computations.
    IS isu, isv;            ///< Vector strides for accessing real and imaginary parts of the state.

  public:
    Gate();

    /**
     * @brief Constructor for quantum gate with specified parameters.
     *
     * @param nlevels_ Number of levels per oscillator.
     * @param nessential_ Number of essential levels per oscillator.
     * @param time_ Final time for gate rotation.
     * @param gate_rot_freq Frequencies of gate rotation (rad/time).
     * @param lindbladtype_ Type of Lindblad operators for open system dynamics.
     * @param quietmode Flag to suppress output messages (default: false).
     */
    Gate(std::vector<int> nlevels_, std::vector<int> nessential_, double time_, std::vector<double> gate_rot_freq, LindbladType lindbladtype_, bool quietmode=false);

    virtual ~Gate();

    /**
     * @brief Retrieves the dimension of the density matrix.
     *
     * @return int Dimension of the system density matrix (all levels).
     */
    int getDimRho() { return dim_rho; };

    /**
     * @brief Assembles the vectorized gate matrices.
     *
     * Computes VxV_re = Re(\bar V \kron V) and VxV_im = Im(\bar V \kron V)
     * where V is the gate matrix and \bar V is its complex conjugate.
     */
    void assembleGate();

    /**
     * @brief Applies the gate transformation to a quantum state.
     *
     * Computes VrhoV = V \rho V^\dagger where V is the gate matrix and \rho is the input state.
     * The output vector VrhoV must be pre-allocated.
     *
     * @param state Input quantum state vector.
     * @param VrhoV Output vector storing the transformed state.
     */
    void applyGate(const Vec state, Vec VrhoV);
};

/**
 * @brief Pauli-X gate implementation.
 *
 * Implements the Pauli-X (NOT) gate for a single qubit:
 * @code
 * V = | 0  1 |
 *     | 1  0 |
 * @endcode
 */
class XGate : public Gate {
  public:
    XGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~XGate();
};

/**
 * @brief Pauli-Y gate implementation.
 *
 * Implements the Pauli-Y gate for a single qubit:
 * @code
 * V = | 0 -i |
 *     | i  0 |
 * @endcode
 */
class YGate : public Gate {
  public:
    YGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~YGate();
};

/**
 * @brief Pauli-Z gate implementation.
 *
 * Implements the Pauli-Z gate for a single qubit:
 * @code
 * V = | 1   0 |
 *     | 0  -1 |
 * @endcode
 */
class ZGate : public Gate {
  public:
    ZGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~ZGate();
};

/**
 * @brief Hadamard gate implementation.
 *
 * Implements the Hadamard gate for a single qubit:
 * @code
 * V = 1/sqrt(2) * | 1   1 |
 *                 | 1  -1 |
 * @endcode
 */
class HadamardGate : public Gate {
  public:
    HadamardGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~HadamardGate();
};

/**
 * @brief Controlled-NOT (CNOT) gate implementation.
 *
 * Implements the CNOT gate for two qubits:
 * @code
 * V = | 1  0  0  0 |
 *     | 0  1  0  0 |
 *     | 0  0  0  1 |
 *     | 0  0  1  0 |
 * @endcode
 */
class CNOT : public Gate {
    public:
    CNOT(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~CNOT();
};

/**
 * @brief SWAP gate implementation.
 *
 * Implements the SWAP gate for two qubits:
 * @code
 * V = | 1  0  0  0 |
 *     | 0  0  1  0 |
 *     | 0  1  0  0 |
 *     | 0  0  0  1 |
 * @endcode
 */
class SWAP: public Gate {
    public:
    SWAP(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~SWAP();
};


/**
 * @brief Multi-qubit SWAP gate implementation.
 *
 * Implements a SWAP gate for Q qubits, swapping qubit 0 with qubit Q-1
 * while leaving all other qubits unchanged.
 */
class SWAP_0Q: public Gate {
    public:
    SWAP_0Q(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~SWAP_0Q();
};

/**
 * @brief Multi-qubit CQNOT gate implementation.
 *
 * Implements a Q-qubit controlled-NOT gate where the NOT operation is applied
 * to qubit Q-1 and is controlled by all other qubits (qubits 0 through Q-2).
 */
class CQNOT: public Gate {
    public:
    CQNOT(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~CQNOT();
};

/**
 * @brief Quantum Fourier Transform gate implementation.
 *
 * Implements the discrete Quantum Fourier Transform gate for multiple qubits.
 */
class QFT: public Gate {
  public:
    QFT(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, bool quietmode=false);
    ~QFT();
};

/**
 * @brief Gate loaded from external file.
 *
 * Implements a custom gate by loading the gate matrix from an external file.
 */
class FromFile: public Gate {
  public:
    /**
     * @brief Constructor for gate loaded from file.
     *
     * @param nlevels_ Number of levels per oscillator.
     * @param nessential_ Number of essential levels per oscillator.
     * @param time Final time for gate rotation.
     * @param rotation_frequencies_ Frequencies of gate rotation.
     * @param lindbladtype_ Type of Lindblad operators.
     * @param filename Path to file containing gate matrix data.
     * @param quietmode Flag to suppress output messages (default: false).
     */
    FromFile(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_, LindbladType lindbladtype_, std::string filename, bool quietmode=false);
    ~FromFile();
};


/**
 * @brief Factory function to initialize a gate from target string.
 *
 * Creates and returns a pointer to the appropriate Gate subclass based on
 * the target specification string.
 *
 * @param target_str Vector of strings specifying the target gate type and parameters.
 * @param nlevels Number of levels per oscillator.
 * @param nessential Number of essential levels per oscillator.
 * @param total_time Total time for gate operation.
 * @param lindbladtype Type of Lindblad operators for open system dynamics.
 * @param gate_rot_freq Frequencies of gate rotation.
 * @param quietmode Flag to suppress output messages.
 * @return Gate* Pointer to the initialized gate object.
 */
Gate* initTargetGate(std::vector<std::string> target_str, std::vector<int>nlevels, std::vector<int>nessential, double total_time, LindbladType lindbladtype, std::vector<double> gate_rot_freq, bool quietmode);
