NUM_PARALLEL_PROCESSORS=0
testNames=(adjoint)
case $subTestNum in
  1)
    rm -rf data_out
    cd ${DIR}/AxC_initBasis0
    $QUANDARY AxC_initBasis0.cfg 
    cd ${DIR}
    ;;
esac
