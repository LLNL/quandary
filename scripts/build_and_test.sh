#!/bin/bash

set -e -x

project_dir="$(pwd)"
hostconfig=${HOST_CONFIG:-""}

# Uberenv
uberenv_cmd="./scripts/uberenv/uberenv.py"
${uberenv_cmd}

# Find cmake cache file (hostconfig)
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "[Error]: No result for: ${project_dir}/*.cmake"
        echo "[Error]: Spack generated host-config not found."
        exit 1
    else
        echo "[Error]: More than one result for: ${project_dir}/*.cmake"
        echo "[Error]: ${hostconfigs[@]}"
        echo "[Error]: Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/${hostconfig}"
fi

hostconfig=$(basename ${hostconfig_path})
echo "[Information]: Found hostconfig ${hostconfig_path}"

# Build
cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`
build_dir="build_${hostconfig//.cmake/}"

rm -rf ${build_dir} 2>/dev/null
mkdir -p ${build_dir} && cd ${build_dir}
$cmake_exe -C ${hostconfig_path} ${project_dir}

make

# Test
ctest

# cd ..
# pytest tests/

