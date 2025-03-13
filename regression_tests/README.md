# Regression Test Documentation

To run tests (in either base directory or `regression_tests/` directory):
```
pytest
```

## Useful options:

- `pytest -v` will print the names of the tests being run

- `pytest -s` will print output to screen

- `pytest -k "AxC_detuning"` runs only the test with name "AxC_detuning"

See `pytest --help` for more options.

## How to add a test

1. Create an appropriate test directory under tests directory, e.g., `mkdir regression_tests/newSimulation`
2. Create a config file in that directory, e.g., `regression_tests/newSimulation/newSimulation.cfg`
3. Add the expected output files of the simulation that you want to test to the base directory, e.g. `regression_tests/newSimulation/base`
4. Add new entry to test_cases.json:
    - The `simulation_name` should be the new directory name, e.g. `newSimulation`.
    - The `files_to_compare` should be an array of output files that should be compared to the expected files in the base directory. You can list them individually or use a regex.
    - The `number_of_processes` is an array of integers. For each integer `i`, a simulation will be run with `mpirun -n ${i}` and the `files_to_compare` will be validated.

## Rebasing tests

To update expected test output for a single simulation, you can do, e.g.:
```
./rebaseTests.sh AxC
```

To update all tests:
```
./rebaseTests.sh
```

## Exact comparison
Normally the tests compare the numerical data of the expected and output files with a relative and absolute tolerance defined in the `regression_test.py`.
To instead do an exact comparison, you can do:
```
pytest --exact
```
