if(ENABLE_TESTS)
  enable_testing()

  blt_add_executable(
    NAME test_configbuilder
    SOURCES test_configbuilder.cpp
    DEPENDS_ON quandary_lib gtest
    FOLDER tests
  )

  blt_add_test(
    NAME test_configbuilder
    COMMAND test_configbuilder
  )
endif()
