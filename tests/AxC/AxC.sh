NUM_PARALLEL_PROCESSORS=0
testNames=(primal)
case $subTestNum in
  1)
    rm -rf data_out
    cd ${DIR}/AxC
    # cp AxCREF.cfg AxC.cfg 
    # sed  "s/DATAOUT/data_out2/g" <AxC.cfg >AxC2.cfg
    # cp AxC2.cfg AxC.cfg
    $QUANDARY AxC.cfg 
    cd ${DIR}
    ;;
esac
