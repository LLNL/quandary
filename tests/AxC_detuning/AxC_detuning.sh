NUM_PARALLEL_PROCESSORS=0
testNames=(adjoint)
case $subTestNum in
  1)
    rm -rf data_out
    cd ${DIR}/AxC_detuning
    $QUANDARY AxC_detuning.cfg 
    cd ${DIR}
    ;;
esac
