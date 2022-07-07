NUM_PARALLEL_PROCESSORS=0
testNames=(adjoint)
case $subTestNum in
  1)
    rm -rf data_out
    cd ${DIR}/AxC_schroedinger_matfree
    $QUANDARY AxC_schroedinger_matfree.cfg 
    cd ${DIR}
    ;;
esac
