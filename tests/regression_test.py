import glob
import json
import os
import subprocess

import pandas as pd
import pytest
from jsonschema import validate

REL_TOL = 1.0e-7
ABS_TOL = 1.0e-15

BASE_DIR = "base"
DATA_OUT_DIR = "data_out"

TEST_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_CASES_PATH = os.path.join(TEST_PATH, "test_cases.json")
TEST_CASE_SCHEMA_PATH = os.path.join(TEST_PATH, "test_case_schema.json")
QUANDARY_PATH = os.path.join(TEST_PATH, "..", "quandary")

def load_test_cases():
    with open(TEST_CASES_PATH) as test_cases_file, open(TEST_CASE_SCHEMA_PATH) as schema_file:
        test_cases = json.load(test_cases_file)
        schema = json.load(schema_file)
        validate(instance=test_cases, schema=schema)
        return test_cases

TEST_CASES = load_test_cases()


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda x: x["simulation_name"])
def test_eval(test_case):
    simulation_name = test_case["simulation_name"]
    files_to_compare = test_case["files_to_compare"]
    number_of_processes_list = test_case["number_of_processes"]

    simulation_dir = os.path.join(TEST_PATH, simulation_name)
    config_file = os.path.join(simulation_dir, simulation_name + ".cfg")

    for number_of_processes in number_of_processes_list:
        run_test(simulation_dir, number_of_processes, config_file, files_to_compare)


def run_test(simulation_dir, number_of_processes, config_file, files_to_compare):
    os.chdir(simulation_dir)
    command = ["mpirun", "-n", str(number_of_processes), QUANDARY_PATH, config_file]
    print(command)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.returncode == 0

    matching_files = [file for pattern in files_to_compare for file in glob.glob(os.path.join(simulation_dir, BASE_DIR, pattern))]
    for expected in matching_files:
        file_name = os.path.basename(expected)
        output = os.path.join(simulation_dir, DATA_OUT_DIR, file_name)
        compare_files(file_name, output, expected)


def compare_files(file_name, output, expected):
        df_output = pd.read_csv(output, sep="\\s+")
        df_expected = pd.read_csv(expected, sep="\\s+")
        pd.testing.assert_frame_equal(df_output, df_expected, rtol=REL_TOL, atol=ABS_TOL, obj=file_name)
