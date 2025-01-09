# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Quandary(CachedCMakePackage):
    """Optimal control for open quantum systems"""

    homepage = "https://github.com/LLNL/quandary"
    git = "https://github.com/LLNL/quandary.git"

    maintainers("steffi7574", "tdrwenski")

    license("MIT", checked_by="tdrwenski")

    version("master", branch="master")

    depends_on("cxx", type="build")

    depends_on("petsc")
    depends_on("petsc+debug", when="+debug")
    depends_on("slepc", when="+slepc")

    variant("slepc", default=False, description="Build with Slepc library")
    variant("debug", default=False, description="Debug mode")

    build_targets = ["all"]
    install_targets = []

    def cmake_args(self):
        args = []
        if '+slepc' in self.spec:
            args.append('-DWITH_SLEPC=ON')
        return args
