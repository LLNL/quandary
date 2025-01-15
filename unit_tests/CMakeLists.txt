blt_add_executable(
    NAME util_tests
    SOURCES util_tests.cpp
    DEPENDS_ON quandary_lib gtest)

blt_add_test(
    NAME util_tests
    COMMAND util_tests)
