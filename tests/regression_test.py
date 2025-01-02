import glob
import pytest
import os
import pandas as pd
import subprocess

from compare_two_files import compare_two_files

REL_TOL = 1.0e-7
ABS_TOL = 1.0e-15

BASE_DIR = "base"
DATA_OUT_DIR = "data_out"

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
QUANDARY = os.path.join(TEST_DIR, "..", "quandary")

BASE_TEST_SET = ["grad.dat", "optim_history.dat"]
EXTENDED_TEST_SET = BASE_TEST_SET + ["rho*.dat", "population*.dat"]

TEST_CASES = [
    ("AxC", EXTENDED_TEST_SET),
    ("AxC_detuning", BASE_TEST_SET),
    ("AxC_groundstate", BASE_TEST_SET),
    ("AxC_initBasis0", BASE_TEST_SET),
    ("AxC_initDiag0", BASE_TEST_SET),
    ("AxC_initFile", BASE_TEST_SET),
    ("AxC_schroedinger_matfree", BASE_TEST_SET),
    ("cnot", EXTENDED_TEST_SET),
    ("cnot-schroedinger", BASE_TEST_SET),
    ("hadamard", BASE_TEST_SET),
    ("pipulse", EXTENDED_TEST_SET),
    ("qubit", BASE_TEST_SET),
    ("xgate", EXTENDED_TEST_SET),
    ("ygate", BASE_TEST_SET),
    ("zgate", BASE_TEST_SET),
]


@pytest.mark.parametrize("simulation_name,files_to_compare", TEST_CASES)
def test_eval(simulation_name, files_to_compare):
    simulation_dir = os.path.join(TEST_DIR, simulation_name)
    config_file = os.path.join(simulation_dir, simulation_name + ".cfg")

    os.chdir(simulation_dir)
    command = [QUANDARY, config_file]
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


def compare_files_old(output, expected):
    compare_two_files(expected, output, REL_TOL, 0)
