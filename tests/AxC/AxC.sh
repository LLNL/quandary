NUM_PARALLEL_PROCESSORS=0
testNames=(adjoint)
case $subTestNum in
  1)
    rm -rf data_out
    cd ${DIR}/AxC
    $QUANDARY AxC.cfg 
    cd ${DIR}
    ;;
esac
