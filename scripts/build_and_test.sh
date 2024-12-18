#!/bin/bash

set -e

echo "~~~~~~~~~~ START:build_and_test.sh ~~~~~~~~~~~"

export PETSC_DIR=/usr/tce/packages/petsc/petsc-3.18.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/lib

make cleanup
make quandary
./tests/runRegressionTests.sh

echo "~~~~~~~~~~ END:build_and_test.sh ~~~~~~~~~~~~~"
