#!/usr/bin/env bash

# Initialize modules for users not using bash as a default shell
if test -e /usr/share/lmod/lmod/init/bash
then
  . /usr/share/lmod/lmod/init/bash
fi

###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Quandary
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
###############################################################################

set -o errexit
set -o nounset

hostname="$(hostname)"
truehostname=${hostname//[0-9]/}
project_dir="$(pwd)"

hostconfig=${HOST_CONFIG:-""}

timed_message ()
{
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~ $(date --rfc-3339=seconds) ~ ${1}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
}

prefix="${project_dir}/../spack-and-build-root"

echo "Creating directory ${prefix}"
echo "project_dir: ${project_dir}"

mkdir -p ${prefix}

spack_cmd="${prefix}/spack/bin/spack"
spack_env_path="${prefix}/spack_env"
uberenv_cmd="./scripts/uberenv/uberenv.py"

# Dependencies
timed_message "Building dependencies"

prefix_opt="--prefix=${prefix}"

# We force Spack to put all generated files (cache and configuration of
# all sorts) in a unique location so that there can be no collision
# with existing or concurrent Spack.
spack_user_cache="${prefix}/spack-user-cache"
export SPACK_DISABLE_LOCAL_CONFIG=""
export SPACK_USER_CACHE_PATH="${spack_user_cache}"
mkdir -p ${spack_user_cache}

${uberenv_cmd} ${prefix_opt}
timed_message "Dependencies built"

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

# Build Directory
build_root=${BUILD_ROOT:-"${prefix}"}

build_dir="${build_root}/build_${hostconfig//.cmake/}"

cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`

# Build
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Prefix: ${prefix}"
echo "~~~~~ Host-config: ${hostconfig_path}"
echo "~~~~~ Build Dir:   ${build_dir}"
echo "~~~~~ Project Dir: ${project_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""
timed_message "Cleaning working directory"

# If building, then delete everything first
rm -rf ${build_dir} 2>/dev/null
mkdir -p ${build_dir} && cd ${build_dir}

timed_message "Building Quandary"
cmake_options=""
if [[ "${truehostname}" == "ruby" || "${truehostname}" == "poodle" ]]
then
    cmake_options="-DBLT_MPI_COMMAND_APPEND:STRING=--overlap"
fi

if [[ "${truehostname}" == "corona" || "${truehostname}" == "tioga" ]]
then
    module unload rocm
fi
$cmake_exe \
    -C ${hostconfig_path} \
    ${cmake_options} \
    ${project_dir}

make

timed_message "Quandary built"

# Test
timed_message "Testing Quandary"
ctest

# cd ..
# pytest tests/
timed_message "Quandary tests completed"
