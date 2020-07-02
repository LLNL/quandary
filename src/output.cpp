#include "output.hpp"

Output::Output(){
  mpirank_world = 0;
  mpirank_petsc = 0;
  mpirank_init  = 0;
  mpirank_braid = 0;
}

Output::Output(MapParam& config, int mpirank_petsc_, int mpirank_init_){
  mpirank_petsc = mpirank_petsc_;
  mpirank_init  = mpirank_init_;
  mpirank_braid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

  /* Create Data directory */
  datadir = config.GetStrParam("datadir", "./data_out");
  if (mpirank_world == 0) {
    mkdir(datadir.c_str(), 0777);
  }
}

Output::Output(MapParam& config, int mpirank_petsc_, int mpirank_init_, int mpirank_braid_) : Output(config, mpirank_petsc_, mpirank_init_) {
  mpirank_braid = mpirank_braid_;
}


Output::~Output(){}