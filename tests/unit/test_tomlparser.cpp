#include <gtest/gtest.h>
#include <sstream>
#include <mpi.h>
#include "config.hpp"

class TomlParserTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize MPI if not already done (for tests)
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      MPI_Init(nullptr, nullptr);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  }

  std::stringstream log;
  int mpi_rank = 0;
};

TEST_F(TomlParserTest, ParseBasicSettings) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    ntime = 500
    dt = 0.05
    collapse_type = "none"
  )", &log, true);

  EXPECT_EQ(config.getNTime(), 500);
  EXPECT_DOUBLE_EQ(config.getDt(), 0.05);
  EXPECT_EQ(config.getCollapseType(), LindbladType::NONE);
}

TEST_F(TomlParserTest, ParseVectorSettings) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 3]
    transfreq = [4.1, 4.8, 5.2]
    rotfreq = [0.0, 0.0]
  )", &log, true);

  auto nlevels = config.getNLevels();
  EXPECT_EQ(nlevels.size(), 2);
  EXPECT_EQ(nlevels[0], 2);
  EXPECT_EQ(nlevels[1], 3);

  auto transfreq = config.getTransFreq();
  EXPECT_EQ(transfreq.size(), 3);
  EXPECT_DOUBLE_EQ(transfreq[0], 4.1);
  EXPECT_DOUBLE_EQ(transfreq[1], 4.8);
  EXPECT_DOUBLE_EQ(transfreq[2], 5.2);
}

TEST_F(TomlParserTest, ParseIndexedSettings) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    control_segments0 = spline, 150
    control_segments1 = step, 1, 2, 3
    output0 = population
    output1 = population
  )", &log, true);

  // Verify control segments were parsed correctly
  EXPECT_EQ(config.getOscillators().size(), 2); // 2 oscillators

  // Check first oscillator
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_segments.size(), 1); // 1 segment
  EXPECT_EQ(osc0.control_segments[0].type, ControlType::BSPLINE);

  // Check second oscillator
  const auto& osc1 = config.getOscillator(1);
  EXPECT_EQ(osc1.control_segments.size(), 1); // 1 segment
  EXPECT_EQ(osc1.control_segments[0].type, ControlType::STEP);

  // Verify output settings
  auto output = config.getOutput();
  EXPECT_EQ(output.size(), 2); // 2 oscillators
  EXPECT_EQ(output[0][0], OutputType::POPULATION);
  EXPECT_EQ(output[1][0], OutputType::POPULATION);
}

TEST_F(TomlParserTest, ParseStructSettings) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    optim_target = gate, cnot
    initialcondition = diagonal, 0
  )", &log, true);

  const auto& target = config.getOptimTarget();
  EXPECT_TRUE(std::holds_alternative<GateOptimTarget>(target));
  const auto& gate_target = std::get<GateOptimTarget>(target);
  EXPECT_EQ(gate_target.gate_type, GateType::CNOT);

  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<DiagonalInitialCondition>(initcond));
  const auto& diag_init = std::get<DiagonalInitialCondition>(initcond);
  EXPECT_EQ(diag_init.osc_IDs, std::vector<size_t>{0});
}

TEST_F(TomlParserTest, ApplyDefaults) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
  )", &log, true);

  // Check defaults were applied
  EXPECT_EQ(config.getNTime(), 1000); // Default ntime
  EXPECT_DOUBLE_EQ(config.getDt(), 0.1); // Default dt
  EXPECT_EQ(config.getCollapseType(), LindbladType::NONE); // Default
}

TEST_F(TomlParserTest, InitialCondition_FromFile) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    initialcondition = file, test.dat
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<FromFileInitialCondition>(initcond));
  EXPECT_EQ(std::get<FromFileInitialCondition>(initcond).filename, "test.dat");
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(TomlParserTest, InitialCondition_Pure) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    initialcondition = pure, 1, 0
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<PureInitialCondition>(initcond));
  const auto& pure_init = std::get<PureInitialCondition>(initcond);
  EXPECT_EQ(pure_init.level_indices, std::vector<size_t>({1,0}));
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(TomlParserTest, InitialCondition_Performance) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    initialcondition = performance
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<PerformanceInitialCondition>(initcond));
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(TomlParserTest, InitialCondition_Ensemble) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    collapse_type = decay
    initialcondition = ensemble, 0, 1
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<EnsembleInitialCondition>(initcond));
  const auto& ensemble_init = std::get<EnsembleInitialCondition>(initcond);
  EXPECT_EQ(ensemble_init.osc_IDs, std::vector<size_t>({0,1}));
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(TomlParserTest, InitialCondition_ThreeStates) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3]
    transfreq = [4.1]
    rotfreq = [0.0]
    collapse_type = decay
    initialcondition = 3states
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<ThreeStatesInitialCondition>(initcond));
  EXPECT_EQ(config.getNInitialConditions(), 3);
}

TEST_F(TomlParserTest, InitialCondition_NPlusOne_SingleOscillator) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3]
    transfreq = [4.1]
    rotfreq = [0.0]
    collapse_type = decay
    initialcondition = nplus1
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<NPlusOneInitialCondition>(initcond));
  // For nlevels = [3], system dimension N = 3, so n_initial_conditions = N + 1 = 4
  EXPECT_EQ(config.getNInitialConditions(), 4);
}

TEST_F(TomlParserTest, InitialCondition_NPlusOne_MultipleOscillators) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 3]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    collapse_type = decay
    initialcondition = nplus1
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<NPlusOneInitialCondition>(initcond));
  // For nlevels = [2, 3], system dimension N = 2 * 3 = 6, so n_initial_conditions = N + 1 = 7
  EXPECT_EQ(config.getNInitialConditions(), 7);
}

TEST_F(TomlParserTest, InitialCondition_Diagonal_Schrodinger) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3, 2]
    nessential = [3, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    collapse_type = none
    initialcondition = diagonal, 1
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<DiagonalInitialCondition>(initcond));
  const auto& diagonal_init = std::get<DiagonalInitialCondition>(initcond);
  EXPECT_EQ(diagonal_init.osc_IDs, std::vector<size_t>({1}));
  // For Schrodinger solver (collapse_type = none), n_initial_conditions = nessential[1] = 2
  EXPECT_EQ(config.getNInitialConditions(), 2);
}

TEST_F(TomlParserTest, InitialCondition_Basis_Schrodinger) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3, 2]
    nessential = [3, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    collapse_type = none
    initialcondition = basis, 1
  )", &log, true);
  // For Schrodinger solver, BASIS is converted to DIAGONAL, so n_initial_conditions = nessential[1] = 2
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<DiagonalInitialCondition>(initcond));
  const auto& diagonal_init = std::get<DiagonalInitialCondition>(initcond);
  EXPECT_EQ(diagonal_init.osc_IDs, std::vector<size_t>({1}));
  EXPECT_EQ(config.getNInitialConditions(), 2);
}

TEST_F(TomlParserTest, InitialCondition_Basis_Lindblad) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3, 2]
    nessential = [3, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    collapse_type = decay
    initialcondition = basis, 1
  )", &log, true);
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<BasisInitialCondition>(initcond));
  const auto& basis_init = std::get<BasisInitialCondition>(initcond);
  EXPECT_EQ(basis_init.osc_IDs, std::vector<size_t>({1}));
  // For Lindblad solver, n_initial_conditions = nessential[1]^2 = 2^2 = 4
  EXPECT_EQ(config.getNInitialConditions(), 4);
}

TEST_F(TomlParserTest, ParsePiPulseSettings_Structure) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2]
    transfreq = [4.1]
    rotfreq = [0.0]
    apply_pipulse = 0, 0.5, 1.0, 0.8
  )", &log, true);

  const auto& pulses = config.getApplyPiPulses();
  EXPECT_EQ(pulses.size(), 2);

  EXPECT_EQ(pulses[0].size(), 1);
  EXPECT_DOUBLE_EQ(pulses[0][0].tstart, 0.5);
  EXPECT_DOUBLE_EQ(pulses[0][0].tstop, 1.0);
  EXPECT_DOUBLE_EQ(pulses[0][0].amp, 0.8);

  // zero pulse for other oscillator
  EXPECT_EQ(pulses[1].size(), 1);
  EXPECT_DOUBLE_EQ(pulses[1][0].tstart, 0.5);
  EXPECT_DOUBLE_EQ(pulses[1][0].tstop, 1.0);
  EXPECT_DOUBLE_EQ(pulses[1][0].amp, 0.0);
}

TEST_F(TomlParserTest, ControlSegments_Spline0) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    control_segments0 = spline0, 150, 0.0, 1.0
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 1);

  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_segments.size(), 1);
  EXPECT_EQ(osc0.control_segments[0].type, ControlType::BSPLINE0);
  SplineParams params0 = std::get<SplineParams>(osc0.control_segments[0].params);
  EXPECT_EQ(params0.nspline, 150);
  EXPECT_DOUBLE_EQ(params0.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params0.tstop, 1.0);
}

TEST_F(TomlParserTest, ControlSegments_Spline) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2]
    transfreq = [4.1, 4.1]
    rotfreq = [0.0, 0.0]
    control_segments0 = spline, 10
    control_segments1 = spline, 20, 0.0, 1.0, spline, 30, 1.0, 2.0
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 2);

  // Check first oscillator with one segment
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_segments.size(), 1);
  EXPECT_EQ(osc0.control_segments[0].type, ControlType::BSPLINE);
  SplineParams params0 = std::get<SplineParams>(osc0.control_segments[0].params);
  EXPECT_EQ(params0.nspline, 10);
  EXPECT_DOUBLE_EQ(params0.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params0.tstop, config.getNTime() * config.getDt());

  // Check second oscillator with two segments
  const auto& osc1 = config.getOscillator(1);
  EXPECT_EQ(osc1.control_segments.size(), 2);

  EXPECT_EQ(osc1.control_segments[0].type, ControlType::BSPLINE);
  SplineParams params1 = std::get<SplineParams>(osc1.control_segments[0].params);
  EXPECT_EQ(params1.nspline, 20);
  EXPECT_DOUBLE_EQ(params1.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params1.tstop, 1.0);

  EXPECT_EQ(osc1.control_segments[1].type, ControlType::BSPLINE);
  SplineParams params2 = std::get<SplineParams>(osc1.control_segments[1].params);
  EXPECT_EQ(params2.nspline, 30);
  EXPECT_DOUBLE_EQ(params2.tstart, 1.0);
  EXPECT_DOUBLE_EQ(params2.tstop, 2.0);
}

TEST_F(TomlParserTest, ControlSegments_Step) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2,2]
    transfreq = [4.1,4.1]
    rotfreq = [0.0,0.0]
    control_segments0 = step, 0.1, 0.2, 0.3, 0.4, 0.5
    control_segments1 = step, 0.1, 0.2, 0.3
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 2);

  // Check first oscillator
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_segments.size(), 1);
  EXPECT_EQ(osc0.control_segments[0].type, ControlType::STEP);
  StepParams params0 = std::get<StepParams>(osc0.control_segments[0].params);
  EXPECT_EQ(params0.step_amp1, 0.1);
  EXPECT_DOUBLE_EQ(params0.step_amp2, 0.2);
  EXPECT_DOUBLE_EQ(params0.tramp, 0.3);
  EXPECT_DOUBLE_EQ(params0.tstart, 0.4);
  EXPECT_DOUBLE_EQ(params0.tstop, 0.5);

  // Check second oscillator
  const auto& osc1 = config.getOscillator(1);
  EXPECT_EQ(osc1.control_segments.size(), 1);
  EXPECT_EQ(osc1.control_segments[0].type, ControlType::STEP);
  StepParams params1 = std::get<StepParams>(osc1.control_segments[0].params);
  EXPECT_EQ(params1.step_amp1, 0.1);
  EXPECT_DOUBLE_EQ(params1.step_amp2, 0.2);
  EXPECT_DOUBLE_EQ(params1.tramp, 0.3);
  EXPECT_DOUBLE_EQ(params1.tstart, 0.0); // default start time
  EXPECT_DOUBLE_EQ(params1.tstop, config.getNTime() * config.getDt()); // default stop time
}

TEST_F(TomlParserTest, ControlSegments_Defaults) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2, 2]
    transfreq = [4.1, 4.8]
    rotfreq = [0.0, 0.0]
    control_segments1 = spline0, 150, 0.0, 1.0
    control_bounds1 = 2.0
  )", &log, true);

  // Verify control segments were parsed correctly
  EXPECT_EQ(config.getOscillators().size(), 3); // 2 oscillators

  // Check first oscillator has default settings
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_segments.size(), 1);
  EXPECT_EQ(osc0.control_segments[0].type, ControlType::BSPLINE);
  SplineParams params0 = std::get<SplineParams>(osc0.control_segments[0].params);
  EXPECT_EQ(params0.nspline, 10);
  EXPECT_DOUBLE_EQ(params0.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params0.tstop, config.getNTime() * config.getDt());

  // Check second oscillator has given settings
  const auto& osc1 = config.getOscillator(1);
  EXPECT_EQ(osc1.control_segments.size(), 1);
  EXPECT_EQ(osc1.control_segments[0].type, ControlType::BSPLINE0);
  EXPECT_EQ(osc1.control_bounds.size(), 1);
  EXPECT_DOUBLE_EQ(osc1.control_bounds[0], 2.0);
  SplineParams params1 = std::get<SplineParams>(osc1.control_segments[0].params);
  EXPECT_EQ(params1.nspline, 150);
  EXPECT_DOUBLE_EQ(params1.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params1.tstop, 1.0);

  // Check third oscillator defaults to the second's settings
  const auto& osc2 = config.getOscillator(2);
  EXPECT_EQ(osc2.control_segments.size(), 1);
  EXPECT_EQ(osc2.control_segments[0].type, ControlType::BSPLINE0);
  SplineParams params2 = std::get<SplineParams>(osc2.control_segments[0].params);
  EXPECT_EQ(params2.nspline, 150);
  EXPECT_DOUBLE_EQ(params2.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params2.tstop, 1.0);
}

TEST_F(TomlParserTest, ControlInitialization_Defaults) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2, 2]
    transfreq = [4.1, 4.1, 4.1]
    rotfreq = [0.0, 0.0, 0.0]
    control_initialization1 = random, 2.0
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 3);

  // Check first oscillator has default settings
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationConstant>(osc0.control_initializations[0]));
  auto params0 = std::get<ControlSegmentInitializationConstant>(osc0.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params0.amplitude, 0.0);
  EXPECT_DOUBLE_EQ(params0.phase, 0.0);

  // Check second oscillator has given settings
  const auto& osc1 = config.getOscillator(1);
  EXPECT_EQ(osc1.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationRandom>(osc1.control_initializations[0]));
  auto params1 = std::get<ControlSegmentInitializationRandom>(osc1.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params1.amplitude, 2.0);
  EXPECT_DOUBLE_EQ(params1.phase, 0.0);

  // Check third oscillator defaults to the second's settings
  const auto& osc2 = config.getOscillator(2);
  EXPECT_EQ(osc2.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationRandom>(osc2.control_initializations[0]));
  auto params2 = std::get<ControlSegmentInitializationRandom>(osc2.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params2.amplitude, 2.0);
  EXPECT_DOUBLE_EQ(params2.phase, 0.0);
}

TEST_F(TomlParserTest, ControlInitialization) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2, 2, 2, 2]
    transfreq = [4.1, 4.1, 4.1, 4.1, 4.1]
    rotfreq = [0.0, 0.0, 0.0, 0.0, 0.0]
    control_initialization0 = constant, 1.0, 1.1
    control_initialization1 = constant, 2.0
    control_initialization2 = random, 3.0, 3.1
    control_initialization3 = random, 4.0
    control_initialization4 = random, 5.0, 5.1, constant, 6.0, 6.1
  )", &log, true);

  // Verify control segments were parsed correctly
  EXPECT_EQ(config.getOscillators().size(), 5);

  // Check first oscillator
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationConstant>(osc0.control_initializations[0]));
  auto params0 = std::get<ControlSegmentInitializationConstant>(osc0.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params0.amplitude, 1.0);
  EXPECT_DOUBLE_EQ(params0.phase, 1.1);

  // Check second oscillator
  const auto& osc1 = config.getOscillator(1);
  EXPECT_EQ(osc1.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationConstant>(osc1.control_initializations[0]));
  auto params1 = std::get<ControlSegmentInitializationConstant>(osc1.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params1.amplitude, 2.0);
  EXPECT_DOUBLE_EQ(params1.phase, 0.0);

  // Check third oscillator
  const auto& osc2 = config.getOscillator(2);
  EXPECT_EQ(osc2.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationRandom>(osc2.control_initializations[0]));
  auto params2 = std::get<ControlSegmentInitializationRandom>(osc2.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params2.amplitude, 3.0);
  EXPECT_DOUBLE_EQ(params2.phase, 3.1);

  // Check fourth oscillator
  const auto& osc3 = config.getOscillator(3);
  EXPECT_EQ(osc3.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationRandom>(osc3.control_initializations[0]));
  auto params3 = std::get<ControlSegmentInitializationRandom>(osc3.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params3.amplitude, 4.0);
  EXPECT_DOUBLE_EQ(params3.phase, 0.0);

  // Check fifth oscillator with two segments
  const auto& osc4 = config.getOscillator(4);
  EXPECT_EQ(osc4.control_initializations.size(), 2);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationRandom>(osc4.control_initializations[0]));
  auto params4_0 = std::get<ControlSegmentInitializationRandom>(osc4.control_initializations[0]);
  EXPECT_DOUBLE_EQ(params4_0.amplitude, 5.0);
  EXPECT_DOUBLE_EQ(params4_0.phase, 5.1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationConstant>(osc4.control_initializations[1]));
  auto params4_1 = std::get<ControlSegmentInitializationConstant>(osc4.control_initializations[1]);
  EXPECT_DOUBLE_EQ(params4_1.amplitude, 6.0);
  EXPECT_DOUBLE_EQ(params4_1.phase, 6.1);
}

TEST_F(TomlParserTest, ControlInitialization_File) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    control_initialization0 = file, params.dat
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 1);
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_initializations.size(), 1);
  EXPECT_TRUE(std::holds_alternative<ControlSegmentInitializationFile>(osc0.control_initializations[0]));
  auto params0 = std::get<ControlSegmentInitializationFile>(osc0.control_initializations[0]);
  EXPECT_EQ(params0.filename, "params.dat");
}

TEST_F(TomlParserTest, ControlBounds) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    control_segments0 = spline, 10, 0.0, 1.0, step, 0.1, 0.2, 0.3, 0.4, 0.5, spline0, 20, 1.0, 2.0
    control_bounds0 = 1.0, 2.0
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 1);

  // Check control bounds for the three segments
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_bounds.size(), 3);
  EXPECT_EQ(osc0.control_bounds[0], 1.0);
  EXPECT_EQ(osc0.control_bounds[1], 2.0);
  EXPECT_EQ(osc0.control_bounds[2], 2.0); // Use last bound for extra segments
}

TEST_F(TomlParserTest, CarrierFrequencies) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    carrier_frequency0 = 1.0, 2.0
  )", &log, true);

  EXPECT_EQ(config.getOscillators().size(), 1);

  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.carrier_frequencies.size(), 2);
  EXPECT_EQ(osc0.carrier_frequencies[0], 1.0);
  EXPECT_EQ(osc0.carrier_frequencies[1], 2.0);
}

TEST_F(TomlParserTest, OptimTarget_GateType) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    optim_target = gate, cnot
  )", &log, true);

  const auto& target = config.getOptimTarget();
  EXPECT_TRUE(std::holds_alternative<GateOptimTarget>(target));
  const auto& gate_target = std::get<GateOptimTarget>(target);
  EXPECT_EQ(gate_target.gate_type, GateType::CNOT);
}

TEST_F(TomlParserTest, OptimTarget_GateFromFile) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    optim_target = gate, file, /path/to/gate.dat
  )", &log, true);

  const auto& target = config.getOptimTarget();
  EXPECT_TRUE(std::holds_alternative<GateOptimTarget>(target));
  const auto& gate_target = std::get<GateOptimTarget>(target);
  EXPECT_EQ(gate_target.gate_type, GateType::FILE);
  EXPECT_EQ(gate_target.gate_file, "/path/to/gate.dat");
}

TEST_F(TomlParserTest, OptimTarget_PureState) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [3, 3, 3]
    transfreq = [4.1]
    rotfreq = [0.0]
    optim_target = pure, 0, 1, 2
  )", &log, true);

  const auto& target = config.getOptimTarget();
  EXPECT_TRUE(std::holds_alternative<PureOptimTarget>(target));
  const auto& pure_target = std::get<PureOptimTarget>(target);
  const auto& levels = pure_target.purestate_levels;
  EXPECT_EQ(levels.size(), 3);
  EXPECT_EQ(levels[0], 0);
  EXPECT_EQ(levels[1], 1);
  EXPECT_EQ(levels[2], 2);
}

TEST_F(TomlParserTest, OptimTarget_FromFile) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
    optim_target = file, /path/to/target.dat
  )", &log, true);

  const auto& target = config.getOptimTarget();
  EXPECT_TRUE(std::holds_alternative<FileOptimTarget>(target));
  const auto& file_target = std::get<FileOptimTarget>(target);
  EXPECT_EQ(file_target.file, "/path/to/target.dat");
}

TEST_F(TomlParserTest, OptimTarget_DefaultPure) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2]
    transfreq = [4.1]
    rotfreq = [0.0]
  )", &log, true);

  const auto& target = config.getOptimTarget();
  EXPECT_TRUE(std::holds_alternative<PureOptimTarget>(target));

  const auto& pure_target = std::get<PureOptimTarget>(target);
  EXPECT_TRUE(pure_target.purestate_levels.empty());
}

TEST_F(TomlParserTest, OptimWeights) {
  Config config = Config::fromTomlString(mpi_rank, R"(
    [system]
    nlevels = [2, 2]
    transfreq = [4.1, 4.1]
    rotfreq = [0.0, 0.0]
    optim_weights = [2.0, 1.0]
  )", &log, true);

  const auto& weights = config.getOptimWeights();
  EXPECT_EQ(weights.size(), 4);
  EXPECT_DOUBLE_EQ(weights[0], 0.4);
  EXPECT_DOUBLE_EQ(weights[1], 0.2);
  EXPECT_DOUBLE_EQ(weights[2], 0.2);
  EXPECT_DOUBLE_EQ(weights[3], 0.2);
}
