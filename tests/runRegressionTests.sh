#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p pdebug
#SBATCH -o sbatch.log
#SBATCH --open-mode truncate

###############################################################################
# SET UP
###############################################################################
usage() {
	echo "Unknown option. Refer to REGRESSIONTEST.md"
	exit 1
}

# Stop at failure
stopAtFailure=false
dryRun=false
rebase=false
tolerance=1.0e-7
isBitWise=0

# Skip setup (git pull, make))
${skipSetup:=false}
# Get options
while getopts ":t:i:e:rh:dh:fh:ph" o;
do
	case "${o}" in
    t)
      tolerance=${OPTARG}
      ;;
		i)
			i=${OPTARG}
      ;;
    e)
      e=${OPTARG}
      ;;
    r)
      rebase=true
      ;;
		d)
			dryRun=true
			;;
		f)
			stopAtFailure=true
			;;
		p)
			isBitWise=1
			;;
    *)
      usage
      ;;
    esac
done
shift $((OPTIND-1))

# If both include and exclude are set, fail
if [[ -n "${i}" ]] && [[ -n "${e}" ]]; then
    usage
		exit 1
fi


# Save directory of this script
if [[ -f "$PWD/runRegressionTests.sh" ]]; then
	DIR=$PWD
elif [[ -f "$PWD/tests/runRegressionTests.sh" ]]; then
	DIR=$PWD/tests
else
	echo "Run tests from the quandary or quandary/tests directory"
fi

# Accumulate tests to run
testsToInclude=( $i )
testsToExclude=( $e )
if [[ ${#testsToInclude[@]} -eq 0 ]];
then
	for file in $DIR/*;
	do
		fileName="$(basename "$file")"

		# Exclude tests from user option
		if [[ " ${testsToExclude[@]} " =~ " ${fileName} " ]]; then
			echo "${file} skipped"
			continue 1
		fi

		# Add test
    if [[ -d "$file" ]] && [[ $fileName != "quandary" ]] && [[ $fileName != "results" ]]; then
			testsToInclude+=($fileName)
    fi
	done
fi

# Check whether tests exist
for simulation in "${testsToInclude[@]}"
do

	# Check if simulation directory exists
	if [ ! -d $DIR/$simulation ]; then
		echo "$simulation test directory doesn't exist. Please try again."
		exit 1
	fi
done

###############################################################################
# SET UP RESULTS DIRECTORY 
###############################################################################

# Create results directory
RESULTS_DIR=$DIR/results

# erase log files in results directory
rm -rf ${RESULTS_DIR}/*.log

###############################################################################
# RUN TESTS
###############################################################################

# Print out which tests will be performed
echo "Running the following tests: "
for simulation in "${testsToInclude[@]}"
do
    echo $simulation
done
echo 

# Check machine
if [ -x "$(command -v srun)" ];
then
    echo 'Using srun -p debug' >&2
    COMMAND="srun -p pdebug"
elif [ -x "$(command -v mpirun)" ];
then
    echo 'Using mpirun -oversubscribe' >&2
    COMMAND="mpirun -oversubscribe"
else
    echo 'ERROR: Neither mpirun nor srun could be found. Exiting.' >%2
    exit 1
fi


# Test number counter
testNum=0
testNumFail=0
testNumPass=0
testNumRebase=0

# Run all tests
for simulation in "${testsToInclude[@]}"
do
	echo "Test $simulation..."

	# Run every script in each test directory
	for script in ${DIR}/${simulation}/*;
	do
		if [[ "$script" == *".sh" ]]; then

      # Get script name without extension
			scriptName=$(basename $script)
      scriptName="${scriptName%.*}"

			subTestNum=0
			parallel=false
			NUM_PARALLEL_PROCESSORS=0

			# Get test names
			. "$script"

			while true;
			do

				if [[ "$parallel" == "false" ]];
				then
					HEADER="$COMMAND -n 1"
					testName="${testNames[$subTestNum]}"
				else
					HEADER="$COMMAND -n $NUM_PARALLEL_PROCESSORS"
					testName="${testNames[$subTestNum]}-parallel"
				fi
        QUANDARY="$HEADER ${DIR}/../quandary"

				# Update subtest numbers
				subTestNum=$((subTestNum+1))

				# Get testtype
				RAN_COMMAND=$(awk "/$subTestNum\)/{f=1;next} /;;/{f=0} f" $script | grep -F '$QUANDARY')
				if [[ $RAN_COMMAND == *"cfg"* ]]; then
					testtype=fom
				else
					if [[ "$parallel" == "false" ]] && [[ $NUM_PARALLEL_PROCESSORS -ne 0 ]];
					then
						parallel=true
						subTestNum=0
						continue
					else
						break
					fi
				fi

				# Update test numbers
				testNum=$((testNum+1))

				# Test failed boolean variable
				testFailed=false

				# Create simulation results log file
				simulationLogFile="${RESULTS_DIR}/${scriptName}-${testName}.log"
				touch $simulationLogFile

				set_pass() {
					testNumPass=$((testNumPass+1))
					if [[ $SLURM == "true" ]]; then
						echo "$testNum. ${scriptName}-${testName}: PASS"
					else
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testName}: PASS"
					fi
					echo "${scriptName}-${testName}: PASS" >> $simulationLogFile
				}

				set_fail() {
					testNumFail=$((testNumFail+1))
					if [[ $SLURM == "true" ]]; then
						echo "$testNum. ${scriptName}-${testName}: FAIL"
					else
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testName}: FAIL"
					fi
					echo "${scriptName}-${testName}: FAIL" >> $simulationLogFile
					if [[ "$stopAtFailure" == "true" ]];
					then
						exit 1
					fi
				}

				set_rebase() {
					testNumRebase=$((testNumRebase+1))
					if [[ $SLURM == "true" ]]; then
						echo "$testNum. ${scriptName}-${testName}: REBASE"
					else
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testName}: REBASE"
					fi
					echo "${scriptName}-${testName}: REBASE" >> $simulationLogFile
				}

				if [[ $SLURM == "false" ]]; then
					echo -n "$testNum. ${scriptName}-${testName}: RUNNING"
				fi

				# Run simulation 
				echo "Running simulation" >> $simulationLogFile 2>&1
				(cd $BASELINE_QUANDARY_DIR && set -o xtrace && . "$script") >> $simulationLogFile 2>&1

				# Check if simulation failed
				if [[ "$?" -ne 0 ]];
				then
					echo "Something went wrong running the baseline simulation with the
					test script: $scriptName. Try running 'make clean' and 'make'." >> $simulationLogFile 2>&1
					set_fail

					# Skip to next test
					continue 1
				fi

				# If doing dry run, skip comparisons
				if [[ "$dryRun" == "false" ]]; then

          if [[ "$rebase" == "true" ]]; then 
            for baseOutput in ${DIR}/${simulation}/data_out/*
            do
              fileName="$(basename "$baseOutput")"
              cd ${DIR}
              if [[ "$fileName" == "grad.dat" ]] || [[ "$fileName" == "optim_history.dat" ]] || [[ "$fileName" == "params.dat" ]]; then
                mv  "${simulation}/data_out/$fileName" "${simulation}/base/$fileName"  
              fi
              if [[ "$testName" == "primal" ]] && [[ "$fileName" == "rho"*".dat" ]]; then
                mv "${simulation}/data_out/$fileName" "${simulation}/base/$fileName"
              fi
              if [[ "$simulation" == "AxC" ]] || [[ "$simulation" == "pipulse" ]] || [[ "$simulation" == "cnot" ]] || [[ "$simulation" == "xgate" ]]; then
                if [[ "$fileName" == "rho"*".dat" ]] || [[ "$fileName" == "population"*".dat" ]]; then
                  mv "${simulation}/data_out/$fileName" "${simulation}/base/$fileName"
                fi
              fi
            done
            set_rebase 
          else
            for baseOutput in ${DIR}/${simulation}/base/*
            do
              fileName="$(basename "$baseOutput")"
              if [[ "$fileName" == "grad.dat" ]] || [[ "$fileName" == "optim_history.dat" ]]; then
                cd ${DIR}
                echo "- comparing $fileName" 
                python3 compare_two_files.py "${simulation}/base/$fileName" "${simulation}/data_out/$fileName" $tolerance $isBitWise
                if [[ $? -eq 1 ]]; then
                  echo "The $baseOutput files are different from the baseline." >> $simulationLogFile 2>&1
                  testFailed=true
                  continue 1
                fi
              fi
              if [[ "${simulation}" == "AxC" ]] || [[ "${simulation}" == "pipulse" ]] || [[ "${simulation}" == "cnot" ]] || [[ "${simulation}" == "xgate" ]]; then
                if [[ "$fileName" == "rho"*".dat" ]] || [[ "$fileName" == "population"*".dat" ]]; then
                  cd ${DIR}
                  echo "- comparing $fileName" 
                  python3 compare_two_files.py "${simulation}/base/$fileName" "${simulation}/data_out/$fileName" $tolerance $isBitWise
                  if [[ $? -eq 1 ]]; then
                    echo "The $baseOutput files are different from the baseline." >> $simulationLogFile 2>&1
                    testFailed=true
                    continue 1
                  fi
                fi
              fi
            done
          fi
				fi

				# Passed
				if [[ "$testFailed" == false ]]; then
					set_pass
        else
          set_fail
				fi
			done
		fi
	done
done


echo "${testNumRebase} rebased, ${testNumPass} passed, ${testNumFail} failed out of ${testNum} tests"
if [[ $testNumFail -ne 0 ]]; then
	exit 1
fi
