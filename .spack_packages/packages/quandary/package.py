# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Quandary(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Optimal control for open quantum systems"""

    homepage = "https://github.com/LLNL/quandary"
    git = "https://github.com/LLNL/quandary.git"

    maintainers("steffi7574", "tdrwenski")

    license("MIT", checked_by="tdrwenski")

    version("master", branch="master")

    depends_on("cxx", type="build")

    depends_on("petsc~hypre~metis~fortran")
    depends_on("petsc~hypre~metis~fortran+debug", when="+debug")
    depends_on("slepc", when="+slepc")

    depends_on("petsc@main", when="+cuda")
    depends_on("petsc@main", when="+rocm")

    variant("slepc", default=False, description="Build with Slepc library")
    variant("debug", default=False, description="Debug mode")

    variant("test", default=False, description="Add dependencies needed for testing")

    with when("+test"):
        depends_on("python", type="run")
        depends_on("py-pip", type="run")
        depends_on("mpi", type="run")

    build_targets = ["all"]
    install_targets = []

    def cmake_args(self):
        args = []
        if '+slepc' in self.spec:
            args.append('-DWITH_SLEPC=ON')
        return args
