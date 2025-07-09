#include "defs.hpp"
#include <vector>
#include <string>
#include <petsc.h>
#include <sstream>

#pragma once

/**
 * @brief Configuration parameter management class with typed member variables.
 *
 * The `Config` class provides a type-safe way to manage Quandary configuration
 * parameters. It supports both programmatic configuration and reading from
 * configuration files for backward compatibility.
 */
class Config {
  public:
    std::stringstream* log; ///< Pointer to log stream for output messages.
    bool quietmode; ///< Flag to control verbose output.

  private:
    std::unordered_map<std::string, std::function<void(const std::string&)>> setters; ///< Setters from config string

    // MPI and logging
    MPI_Comm comm; ///< MPI communicator for parallel operations.
    int mpi_rank; ///< MPI rank of the current process.

    // General options
    std::vector<int> nlevels;  ///< Number of levels per subsystem
    std::vector<int> nessential;  ///< Number of essential levels per subsystem (Default: same as nlevels)
    int ntime = 1000;  ///< Number of time steps used for time-integration
    double dt = 0.1;  ///< Time step size (ns). Determines final time: T=ntime*dt
    std::vector<double> transfreq;  ///< Fundamental transition frequencies for each oscillator (GHz)
    std::vector<double> selfkerr;  ///< Self-kerr frequencies for each oscillator (GHz)
    std::vector<double> crosskerr;  ///< Cross-kerr coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> Jkl;  ///< Dipole-dipole coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> rotfreq;  ///< Rotational wave approximation frequencies for each subsystem (GHz)
    LindbladType collapse_type = LindbladType::NONE;  ///< Switch between Schroedinger and Lindblad solver
    std::vector<double> decay_time;  ///< Time of decay collapse operation (T1) per oscillator (for Lindblad solver)
    std::vector<double> dephase_time;  ///< Time of dephase collapse operation (T2) per oscillator (for Lindblad solver)
    InitialConditionType initialcondition = InitialConditionType::BASIS;  ///< Specify the initial conditions that are to be propagated
    std::vector<double> apply_pipulse;  ///< Apply a pi-pulse to oscillator with specified parameters

    // Optimization options
    std::vector<std::string> control_segments;  ///< Define the control segments for each oscillator
    bool control_enforceBC = false;  ///< Decide whether control pulses should start and end at zero
    std::vector<std::string> control_initialization;  ///< Set the initial control pulse parameters for each oscillator
    std::vector<double> control_bounds;  ///< Maximum amplitude bound for the control pulses for each oscillator (GHz)
    std::vector<std::vector<double>> carrier_frequencies;  ///< Carrier wave frequencies for each oscillator (GHz)
    std::vector<TargetType> optim_target = {TargetType::PURE};  ///< Optimization target
    std::vector<double> gate_rot_freq;  ///< Frequency of rotation of the target gate, for each oscillator (GHz)
    ObjectiveType optim_objective = ObjectiveType::JFROBENIUS;  ///< Objective function measure
    std::vector<double> optim_weights;  ///< Weights for summing up the objective function
    double optim_atol = 1e-8;  ///< Optimization stopping tolerance based on gradient norm (absolute)
    double optim_rtol = 1e-4;  ///< Optimization stopping tolerance based on gradient norm (relative)
    double optim_ftol = 1e-8;  ///< Optimization stopping criterion based on the final time cost (absolute)
    double optim_inftol = 1e-5;  ///< Optimization stopping criterion based on the infidelity (absolute)
    int optim_maxiter = 200;  ///< Maximum number of optimization iterations
    double optim_regul = 1e-4;  ///< Coefficient of Tikhonov regularization for the design variables
    double optim_penalty = 0.0;  ///< Coefficient for adding first integral penalty term
    double optim_penalty_param = 0.5;  ///< Integral penalty parameter inside the weight (gaussian variance a)
    double optim_penalty_dpdm = 0.0;  ///< Coefficient for penalizing the integral of the second derivative of state populations
    double optim_penalty_energy = 0.0;  ///< Coefficient for penalizing the control pulse energy integral
    double optim_penalty_variation = 0.01;  ///< Coefficient for penalizing variations in control amplitudes
    // bool optim_regul_interpolate = false;  ///< TODO this isnt in config_template.cfg, use optim_regul_tik0?
    bool optim_regul_tik0 = false;  ///< Switch to use Tikhonov regularization with ||x - x_0||^2 instead of ||x||^2 TODO this isnt used

    // Output and runtypes
    std::string datadir = "./data_out";  ///< Directory for output files
    std::vector<std::vector<std::string>> output;  ///< Specify the desired output for each oscillator
    int output_frequency = 1;  ///< Output frequency in the time domain: write output every <num> time-step
    int optim_monitor_frequency = 10;  ///< Frequency of writing output during optimization iterations
    RunType runtype = RunType::SIMULATION;  ///< Runtype options: simulation, gradient, or optimization
    bool usematfree = false;  ///< Use matrix free solver, instead of sparse matrix implementation
    std::string linearsolver_type = "gmres";  ///< Solver type for solving the linear system at each time step
    int linearsolver_maxiter = 10;  ///< Set maximum number of iterations for the linear solver
    std::string timestepper = "IMR";  ///< Switch the time-stepping algorithm (IMR, IMR4, IMR8)
    int rand_seed;  ///< Fixed seed for the random number generator for reproducability

  public:
    // Constructors
    Config();
    Config(MPI_Comm comm_, std::stringstream& logstream, bool quietmode=false);
    ~Config();

    static Config createFromFile(const std::string& filename, MPI_Comm comm, std::stringstream& logstream, bool quietmode = false);
    void loadFromFile(const std::string& filename);
    void applyConfigLine(const std::string& line);
    void printConfig() const;

    // getters
    const std::vector<int>& getNLevels() const { return nlevels; }
    const std::vector<int>& getNEssential() const { return nessential; }
    int getNTime() const { return ntime; }
    double getDt() const { return dt; }
    const std::vector<double>& getTransFreq() const { return transfreq; }
    const std::vector<double>& getSelfKerr() const { return selfkerr; }
    const std::vector<double>& getCrossKerr() const { return crosskerr; }
    const std::vector<double>& getJkl() const { return Jkl; }
    const std::vector<double>& getRotFreq() const { return rotfreq; }
    LindbladType getCollapseType() const { return collapse_type; }
    const std::vector<double>& getDecayTime() const { return decay_time; }
    const std::vector<double>& getDephaseTime() const { return dephase_time; }
    InitialConditionType getInitialCondition() const { return initialcondition; }
    const std::vector<double>& getApplyPiPulse() const { return apply_pipulse; }

    const std::vector<std::string>& getControlSegments() const { return control_segments; }
    bool getControlEnforceBC() const { return control_enforceBC; }
    const std::vector<std::string>& getControlInitialization() const { return control_initialization; }
    const std::vector<double>& getControlBounds() const { return control_bounds; }
    const std::vector<std::vector<double>>& getCarrierFrequency0() const { return carrier_frequencies; }
    const std::vector<TargetType>& getOptimTarget() const { return optim_target; }
    const std::vector<double>& getGateRotFreq() const { return gate_rot_freq; }
    ObjectiveType getOptimObjective() const { return optim_objective; }
    const std::vector<double>& getOptimWeights() const { return optim_weights; }
    double getOptimAtol() const { return optim_atol; }
    double getOptimRtol() const { return optim_rtol; }
    double getOptimFtol() const { return optim_ftol; }
    double getOptimInftol() const { return optim_inftol; }
    int getOptimMaxiter() const { return optim_maxiter; }
    double getOptimRegul() const { return optim_regul; }
    double getOptimPenalty() const { return optim_penalty; }
    double getOptimPenaltyParam() const { return optim_penalty_param; }
    double getOptimPenaltyDpdm() const { return optim_penalty_dpdm; }
    double getOptimPenaltyEnergy() const { return optim_penalty_energy; }
    double getOptimPenaltyVariation() const { return optim_penalty_variation; }
    bool getOptimRegulTik0() const { return optim_regul_tik0; }

    const std::string& getDataDir() const { return datadir; }
    const std::vector<std::vector<std::string>>& getOutput() const { return output; }
    int getOutputFrequency() const { return output_frequency; }
    int getOptimMonitorFrequency() const { return optim_monitor_frequency; }
    RunType getRuntype() const { return runtype; }
    bool getUseMatFree() const { return usematfree; }
    const std::string& getLinearSolverType() const { return linearsolver_type; }
    int getLinearSolverMaxiter() const { return linearsolver_maxiter; }
    const std::string& getTimestepper() const { return timestepper; }
    int getRandSeed() const { return rand_seed; }

    // setters
    void setNLevels(const std::vector<int>& nelevels_);
    void setRandSeed(int rand_seed_);

  private:
    std::vector<std::string> split(const std::string& str, char delimiter = ',');

    template<typename T>
    T convertFromString(const std::string& str) {
      return str;
    }

    template<>
    int convertFromString<int>(const std::string& str) {
      return std::stoi(str);
    }

    template<>
    double convertFromString<double>(const std::string& str) {
      return std::stod(str);
    }

    template<typename T>
    void registerScalar(const std::string& key, T& member, void (Config::*setter)(T) = nullptr) {
      setters[key] = [this, &member, setter](const std::string& val) {
        T converted_val = convertFromString<T>(val);

        if (setter != nullptr) {
          (this->*setter)(converted_val);
        } else {
          member = converted_val;
        }
      };
    }

    template<typename T>
    void registerVector(const std::string& key, std::vector<T>& member) {
        setters[key] = [this, &member](const std::string& val) {
            auto parts = split(val);
            member.clear();
            for (const auto& part : parts) {
                member.push_back(convertFromString<T>(part));
            }
        };
    }
  };