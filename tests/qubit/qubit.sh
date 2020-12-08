NUM_PARALLEL_PROCESSORS=0
testNames=(primal)
case $subTestNum in
  1)
    rm -rf data_out
    cd ${DIR}/qubit
    # cp qubitREF.cfg qubit.cfg 
    # sed  "s/DATAOUT/data_out2/g" <qubit.cfg >qubit2.cfg
    # cp qubit2.cfg qubit.cfg
    $QUANDARY qubit.cfg 
    cd ${DIR}
    ;;
esac
