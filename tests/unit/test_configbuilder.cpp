#include <gtest/gtest.h>
#include <sstream>
#include <mpi.h>
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
    control_segments0 = spline, 150
    control_segments1 = step, 10
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
  )");

  Config config = builder.build();

  // Check defaults were applied
  EXPECT_EQ(config.getNTime(), 1000); // Default ntime
  EXPECT_DOUBLE_EQ(config.getDt(), 0.1); // Default dt
  EXPECT_EQ(config.getCollapseType(), LindbladType::NONE); // Default
}
