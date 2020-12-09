# Regression Test Documentation

The script will perform regression test  


## How to run the tests on LC

1. sbatch tests/runRegressionTests.sh (if in the base directory) or sbatch runRegressionTests.sh (if in the tests directory). Look below for test options.
2. The slurm output file will be stored in sbatch.log in the directory you ran the previous command from.
3. Test commands/logs are stored in tests/results. The output data of each test case is stored in tests/[test case]/data_out.
4. To find the particular error and where it occurred, scroll to the bottom of each test file in tests/results. You can then compare tests/[test case]/datat_out/[data file] with tests/[test case]/base/[data file] 
5. To erase the regression test data, run from the base directory: make clean-regtest

## How to run the tests on MAC

1. ./tests/runRegressionTests.sh (if in the base directory) or ./runRegressionTests.sh (if in the tests directory). Look below for test options.
2. Follow steps 3-5 from the instructions above.

## How to add a test 

1. Create an appropriate test directory under tests directory, e.g., mkdir tests/AxCpiPulse
2. Copy qubit/qubit.sh to your test directory, renaming it as desired, e.g., cp qubit/qubit.sh AxCpiPulse/
3. Choose a number for NUM_PARALLEL_PROCESSORS in your sh file.
4. Name your tests in testNames.
5. Create an appropriate input file in your test directory and name it same as the test directory, but with .cfg extension.

## Here are some example runs and results:

./runRegressionTests.sh -> Run all tests.

./runRegressionTests.sh -f -> Run all tests, stopping at the first test failure on each processor.

./runRegressionTests.sh -d -> Run all tests, but do not do comparisons. Failures only occur if the simulations do not run successfully.

./runRegressionTests.sh -i  "AxC qubit" -> Run AxC and qubit.

./runRegressionTests.sh -i "qubit" -r -> Run qubit and rebase the reference solution of qubit.

./runRegressionTests.sh -e "AxC" -> Run all tests except AxC 

./runRegressionTests.sh -i "AxC" -e "qubit" -> Error. -i and -e can not be used simultaneously.
