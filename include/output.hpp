#include <sys/stat.h> 
#include <iostream> 
#include "config.hpp"
#pragma once


class Output{

  int mpirank_world;  /* Rank of processor of MPI_COMM_WORLD */
  int mpirank_petsc;  /* Rank of processor for parallelizing Petsc */
  int mpirank_init;   /* Rank of processor for parallelizing initial conditions */
  int mpirank_braid;  /* Rank of processor for parallelizing XBraid, or zero if compiling without XBraid */

  public:
    std::string datadir;

  public:
    Output();
    Output(MapParam& config, int mpirank_petsc, int mpirank_init);
    Output(MapParam& config, int mpirank_petsc, int mpirank_init, int mpirank_braid);
    ~Output();
};