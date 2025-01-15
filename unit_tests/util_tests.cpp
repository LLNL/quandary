#include <gtest/gtest.h>

#include "util.hpp"

const double TOL = 1e-6;

TEST(Util, test_sigmoid)
{
    ASSERT_NEAR(0.5, sigmoid(0.0, 1.0), TOL);
    ASSERT_NEAR(0.5, sigmoid(1.0, 0.0), TOL);
    ASSERT_NEAR(1.0, sigmoid(1.0, 100.0), TOL);
    ASSERT_NEAR(0.0, sigmoid(1.0, -100.0), TOL);
}
