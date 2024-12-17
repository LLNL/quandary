#!/bin/bash

set -e

export PETSC_DIR=/usr/tce/packages/petsc/petsc-3.18.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/lib

make cleanup
make quandary
./tests/runRegressionTests.sh
