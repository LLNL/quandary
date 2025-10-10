#include "config.hpp"
#include <iostream>
#include <string>

#include "configbuilder.hpp"

Config::Config(
  MPI_Comm comm_,
  // System parameters
  const std::vector<size_t>& nlevels_,
  const std::vector<size_t>& nessential_,
  int ntime_,
  double dt_,
  const std::vector<double>& transfreq_,
  const std::vector<double>& selfkerr_,
  const std::vector<double>& crosskerr_,
  const std::vector<double>& Jkl_,
  const std::vector<double>& rotfreq_,
  LindbladType collapse_type_,
  const std::vector<double>& decay_time_,
  const std::vector<double>& dephase_time_,
  InitialConditionType initial_condition_type_,
  int n_initial_conditions_,
  const std::vector<size_t>& initial_condition_IDs_,
  const std::string& initial_condition_file_,
  const std::vector<std::vector<PiPulseSegment>>& apply_pipulse_,
  // Control parameters
  const std::vector<std::vector<ControlSegment>>& control_segments_,
  bool control_enforceBC_,
  const std::vector<std::vector<ControlSegmentInitialization>>& control_initializations_,
  const std::optional<std::string>& control_initialization_file_,
  const std::vector<std::vector<double>>& control_bounds_,
  const std::vector<std::vector<double>>& carrier_frequencies_,
  // Optimization parameters
  TargetType optim_target_type_,
  const std::string& optim_target_file_,
  GateType optim_target_gate_type_,
  const std::string& optim_target_gate_file_,
  const std::vector<size_t>& optim_target_purestate_levels_,
  const std::vector<double>& gate_rot_freq_,
  ObjectiveType optim_objective_,
  const std::vector<double>& optim_weights_,
  double optim_atol_,
  double optim_rtol_,
  double optim_ftol_,
  double optim_inftol_,
  int optim_maxiter_,
  double optim_regul_,
  double optim_penalty_,
  double optim_penalty_param_,
  double optim_penalty_dpdm_,
  double optim_penalty_energy_,
  double optim_penalty_variation_,
  bool optim_regul_tik0_,
  // Output parameters
  const std::string& datadir_,
  const std::vector<std::vector<OutputType>>& output_,
  int output_frequency_,
  int optim_monitor_frequency_,
  RunType runtype_,
  bool usematfree_,
  LinearSolverType linearsolver_type_,
  int linearsolver_maxiter_,
  TimeStepperType timestepper_type_,
  int rand_seed_,
  const std::string& hamiltonian_file_Hsys_,
  const std::string& hamiltonian_file_Hc_
) :
  comm(comm_),
  nlevels(nlevels_),
  nessential(nessential_),
  ntime(ntime_),
  dt(dt_),
  transfreq(transfreq_),
  selfkerr(selfkerr_),
  crosskerr(crosskerr_),
  Jkl(Jkl_),
  rotfreq(rotfreq_),
  collapse_type(collapse_type_),
  decay_time(decay_time_),
  dephase_time(dephase_time_),
  initial_condition_type(initial_condition_type_),
  n_initial_conditions(n_initial_conditions_),
  initial_condition_IDs(initial_condition_IDs_),
  initial_condition_file(initial_condition_file_),
  apply_pipulse(apply_pipulse_),
  control_segments(control_segments_),
  control_enforceBC(control_enforceBC_),
  control_initializations(control_initializations_),
  control_initialization_file(control_initialization_file_),
  control_bounds(control_bounds_),
  carrier_frequencies(carrier_frequencies_),
  optim_target_type(optim_target_type_),
  optim_target_file(optim_target_file_),
  optim_target_gate_type(optim_target_gate_type_),
  optim_target_gate_file(optim_target_gate_file_),
  optim_target_purestate_levels(optim_target_purestate_levels_),
  gate_rot_freq(gate_rot_freq_),
  optim_objective(optim_objective_),
  optim_weights(optim_weights_),
  optim_atol(optim_atol_),
  optim_rtol(optim_rtol_),
  optim_ftol(optim_ftol_),
  optim_inftol(optim_inftol_),
  optim_maxiter(optim_maxiter_),
  optim_regul(optim_regul_),
  optim_penalty(optim_penalty_),
  optim_penalty_param(optim_penalty_param_),
  optim_penalty_dpdm(optim_penalty_dpdm_),
  optim_penalty_energy(optim_penalty_energy_),
  optim_penalty_variation(optim_penalty_variation_),
  optim_regul_tik0(optim_regul_tik0_),
  datadir(datadir_),
  output(output_),
  output_frequency(output_frequency_),
  optim_monitor_frequency(optim_monitor_frequency_),
  runtype(runtype_),
  usematfree(usematfree_),
  linearsolver_type(linearsolver_type_),
  linearsolver_maxiter(linearsolver_maxiter_),
  timestepper_type(timestepper_type_),
  rand_seed(rand_seed_),
  hamiltonian_file_Hsys(hamiltonian_file_Hsys_),
  hamiltonian_file_Hc(hamiltonian_file_Hc_)
{
  MPI_Comm_rank(comm, &mpi_rank);
}

Config::~Config(){}

Config Config::fromCfg(std::string filename, std::stringstream* log, bool quietmode) {
  ConfigBuilder builder(MPI_COMM_WORLD, *log, quietmode);
  builder.loadFromFile(filename);
  return builder.build();
}

void Config::printConfig() const {
  std::cout << "Configuration:\n";

  std::cout << "  nlevels = ";
  for (size_t i = 0; i < nlevels.size(); ++i) {
    std::cout << nlevels[i];
    if (i < nlevels.size() - 1) std::cout << ", ";
  }
  std::cout << "\n";

  std::cout << "  nessential = ";
  for (size_t i = 0; i < nessential.size(); ++i) {
    std::cout << nessential[i];
    if (i < nessential.size() - 1) std::cout << ", ";
  }
  std::cout << "\n";

  std::cout << "  ntime = " << ntime << "\n";
  std::cout << "  dt = " << dt << "\n";
  std::cout << "\n";
}
