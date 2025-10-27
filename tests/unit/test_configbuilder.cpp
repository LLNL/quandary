#include <cstdio>
#include <gtest/gtest.h>
#include <sstream>
#include <mpi.h>
#include "config.hpp"
#include "configbuilder.hpp"

class ConfigBuilderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize MPI if not already done (for tests)
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      MPI_Init(nullptr, nullptr);
    }
  }

  std::stringstream log;
};

TEST_F(ConfigBuilderTest, ParseBasicSettings) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test basic setting parsing
  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
    ntime = 500
    dt = 0.05
    collapse_type = none
  )");

  Config config = builder.build();

  EXPECT_EQ(config.getNTime(), 500);
  EXPECT_DOUBLE_EQ(config.getDt(), 0.05);
  EXPECT_EQ(config.getCollapseType(), LindbladType::NONE);
}

TEST_F(ConfigBuilderTest, ParseVectorSettings) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test vector parsing
  builder.loadFromString(R"(
    nlevels = 2, 3
    transfreq = 4.1, 4.8, 5.2
    rotfreq = 0.0, 0.0
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ParseIndexedSettings) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test indexed setting parsing
  builder.loadFromString(R"(
    nlevels = 2, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    control_segments0 = spline, 150
    control_segments1 = step, 1, 2, 3
    output0 = population
    output1 = population
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ParseStructSettings) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test struct parsing (multi-parameter settings)
  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
    optim_target = gate, cnot
    initialcondition = diagonal, 0
  )");

  Config config = builder.build();

  EXPECT_EQ(config.getOptimTargetType(), TargetType::GATE);
  EXPECT_EQ(config.getOptimTargetGateType(), GateType::CNOT);

  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<DiagonalInitialCondition>(initcond));
  const auto& diag_init = std::get<DiagonalInitialCondition>(initcond);
  EXPECT_EQ(diag_init.osc_IDs, std::vector<size_t>{0});
}

TEST_F(ConfigBuilderTest, ApplyDefaults) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Provide minimal config, test that defaults are applied
  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
  )");

  Config config = builder.build();

  // Check defaults were applied
  EXPECT_EQ(config.getNTime(), 1000); // Default ntime
  EXPECT_DOUBLE_EQ(config.getDt(), 0.1); // Default dt
  EXPECT_EQ(config.getCollapseType(), LindbladType::NONE); // Default
}

TEST_F(ConfigBuilderTest, InitialCondition_FromFile) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
    initialcondition = file, test.dat
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<FromFileInitialCondition>(initcond));
  EXPECT_EQ(std::get<FromFileInitialCondition>(initcond).filename, "test.dat");
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(ConfigBuilderTest, InitialCondition_Pure) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 3, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    initialcondition = pure, 1, 0
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<PureInitialCondition>(initcond));
  const auto& pure_init = std::get<PureInitialCondition>(initcond);
  EXPECT_EQ(pure_init.level_indices, std::vector<size_t>({1,0}));
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(ConfigBuilderTest, InitialCondition_Performance) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
    initialcondition = performance
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<PerformanceInitialCondition>(initcond));
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(ConfigBuilderTest, InitialCondition_Ensemble) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 3, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    collapse_type = decay
    initialcondition = ensemble, 0, 1
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<EnsembleInitialCondition>(initcond));
  const auto& ensemble_init = std::get<EnsembleInitialCondition>(initcond);
  EXPECT_EQ(ensemble_init.osc_IDs, std::vector<size_t>({0,1}));
  EXPECT_EQ(config.getNInitialConditions(), 1);
}

TEST_F(ConfigBuilderTest, InitialCondition_ThreeStates) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);
  builder.loadFromString(R"(
    nlevels = 3
    transfreq = 4.1
    rotfreq = 0.0
    collapse_type = decay
    initialcondition = 3states
  )");
  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<ThreeStatesInitialCondition>(initcond));
  EXPECT_EQ(config.getNInitialConditions(), 3);
}

TEST_F(ConfigBuilderTest, InitialCondition_NPlusOne_SingleOscillator) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 3
    transfreq = 4.1
    rotfreq = 0.0
    collapse_type = decay
    initialcondition = nplus1
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<NPlusOneInitialCondition>(initcond));
  // For nlevels = [3], system dimension N = 3, so n_initial_conditions = N + 1 = 4
  EXPECT_EQ(config.getNInitialConditions(), 4);
}

TEST_F(ConfigBuilderTest, InitialCondition_NPlusOne_MultipleOscillators) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 2, 3
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    collapse_type = decay
    initialcondition = nplus1
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<NPlusOneInitialCondition>(initcond));
  // For nlevels = [2, 3], system dimension N = 2 * 3 = 6, so n_initial_conditions = N + 1 = 7
  EXPECT_EQ(config.getNInitialConditions(), 7);
}

TEST_F(ConfigBuilderTest, InitialCondition_Diagonal_Schrodinger) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 3, 2
    nessential = 3, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    collapse_type = none
    initialcondition = diagonal, 1
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<DiagonalInitialCondition>(initcond));
  const auto& diagonal_init = std::get<DiagonalInitialCondition>(initcond);
  EXPECT_EQ(diagonal_init.osc_IDs, std::vector<size_t>({1}));
  // For Schrodinger solver (collapse_type = none), n_initial_conditions = nessential[1] = 2
  EXPECT_EQ(config.getNInitialConditions(), 2);
}

TEST_F(ConfigBuilderTest, InitialCondition_Basis_Schrodinger) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 3, 2
    nessential = 3, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    collapse_type = none
    initialcondition = basis, 1
  )");

  Config config = builder.build();
  // For Schrodinger solver, BASIS is converted to DIAGONAL, so n_initial_conditions = nessential[1] = 2
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<DiagonalInitialCondition>(initcond));
  const auto& diagonal_init = std::get<DiagonalInitialCondition>(initcond);
  EXPECT_EQ(diagonal_init.osc_IDs, std::vector<size_t>({1}));
  EXPECT_EQ(config.getNInitialConditions(), 2);
}

TEST_F(ConfigBuilderTest, InitialCondition_Basis_Lindblad) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 3, 2
    nessential = 3, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    collapse_type = decay
    initialcondition = basis, 1
  )");

  Config config = builder.build();
  const auto& initcond = config.getInitialCondition();
  EXPECT_TRUE(std::holds_alternative<BasisInitialCondition>(initcond));
  const auto& basis_init = std::get<BasisInitialCondition>(initcond);
  EXPECT_EQ(basis_init.osc_IDs, std::vector<size_t>({1}));
  // For Lindblad solver, n_initial_conditions = nessential[1] = 2
  EXPECT_EQ(config.getNInitialConditions(), 2);
}

TEST_F(ConfigBuilderTest, ParsePiPulseSettings_Structure) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test basic pi-pulse data structure initialization
  builder.loadFromString(R"(
    nlevels = 2, 2
    transfreq = 4.1
    rotfreq = 0.0
    apply_pipulse = 0, 0.5, 1.0, 0.8
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ControlSegments_Spline0) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test control segment parsing
  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
    control_segments0 = spline0, 150, 0.0, 1.0
  )");

  Config config = builder.build();

  EXPECT_EQ(config.getOscillators().size(), 1);

  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_segments.size(), 1);
  EXPECT_EQ(osc0.control_segments[0].type, ControlType::BSPLINE0);
  SplineParams params0 = std::get<SplineParams>(osc0.control_segments[0].params);
  EXPECT_EQ(params0.nspline, 150);
  EXPECT_DOUBLE_EQ(params0.tstart, 0.0);
  EXPECT_DOUBLE_EQ(params0.tstop, 1.0);
}

TEST_F(ConfigBuilderTest, ControlSegments_Spline) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test control segment parsing
  builder.loadFromString(R"(
    nlevels = 2, 2
    transfreq = 4.1, 4.1
    rotfreq = 0.0, 0.0
    control_segments0 = spline, 10
    control_segments1 = spline, 20, 0.0, 1.0, spline, 30, 1.0, 2.0
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ControlSegments_Step) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test control segment parsing
  builder.loadFromString(R"(
    nlevels = 2,2
    transfreq = 4.1,4.1
    rotfreq = 0.0,0.0
    control_segments0 = step, 0.1, 0.2, 0.3, 0.4, 0.5
    control_segments1 = step, 0.1, 0.2, 0.3
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ControlSegments_Defaults) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  // Test control segment parsing
  builder.loadFromString(R"(
    nlevels = 2, 2, 2
    transfreq = 4.1, 4.8
    rotfreq = 0.0, 0.0
    control_segments1 = spline0, 150, 0.0, 1.0
    control_bounds1 = 2.0
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ControlInitialization_Defaults) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 2, 2, 2
    transfreq = 4.1, 4.1, 4.1
    rotfreq = 0.0, 0.0, 0.0
    control_initialization1 = random, 2.0
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ControlInitialization) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 2, 2, 2, 2, 2
    transfreq = 4.1, 4.1, 4.1, 4.1, 4.1
    rotfreq = 0.0, 0.0, 0.0, 0.0, 0.0
    control_initialization0 = constant, 1.0, 1.1
    control_initialization1 = constant, 2.0
    control_initialization2 = random, 3.0, 3.1
    control_initialization3 = random, 4.0
    control_initialization4 = random, 5.0, 5.1, constant, 6.0, 6.1
  )");

  Config config = builder.build();

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

TEST_F(ConfigBuilderTest, ControlBounds) {
  ConfigBuilder builder(MPI_COMM_WORLD, log, true);

  builder.loadFromString(R"(
    nlevels = 2
    transfreq = 4.1
    rotfreq = 0.0
    control_segments0 = spline, 10, 0.0, 1.0, step, 0.1, 0.2, 0.3, 0.4, 0.5, spline0, 20, 1.0, 2.0
    control_bounds0 = 1.0, 2.0
  )");

  Config config = builder.build();

  EXPECT_EQ(config.getOscillators().size(), 1);

  // Check control bounds for the three segments
  const auto& osc0 = config.getOscillator(0);
  EXPECT_EQ(osc0.control_bounds.size(), 3);
  EXPECT_EQ(osc0.control_bounds[0], 1.0);
  EXPECT_EQ(osc0.control_bounds[1], 2.0);
  EXPECT_EQ(osc0.control_bounds[2], 2.0); // Use last bound for extra segments
}
