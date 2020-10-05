#include "output.hpp"

Output::Output(){
  mpirank_world = 0;
  mpirank_petsc = 0;
  mpirank_init  = 0;
  mpirank_braid = 0;
  optim_monitor_freq = 0;
  optim_iter = 0;
}

Output::Output(MapParam& config, int mpirank_petsc_, int mpirank_init_) : Output() {
  mpirank_petsc = mpirank_petsc_;
  mpirank_init  = mpirank_init_;
  mpirank_braid = 0;

  /* Get rank of global world communicator */
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

  /* Create Data directory */
  datadir = config.GetStrParam("datadir", "./data_out");
  if (mpirank_world == 0) {
    mkdir(datadir.c_str(), 0777);
  }

  /* Prepare output for optimizer */
  optim_monitor_freq = config.GetIntParam("optim_monitor_frequency", 10);
  if (mpirank_world == 0) {
    char filename[255];
    sprintf(filename, "%s/optim_history.dat", datadir.c_str());
    optimfile = fopen(filename, "w");
    fprintf(optimfile, "#iter    obj_value           ||grad||               LS step           Costfunction     Tikhonov-regul      Penalty-term\n");
  } 

  /* Read from config what output is desired */
  std::vector<int> nlevels;
  config.GetVecIntParam("nlevels", nlevels);
  for (int i = 0; i < nlevels.size(); i++){
    std::vector<std::string> fillme;
    config.GetVecStrParam("output" + std::to_string(i), fillme, "none");
    outputstr.push_back(fillme);
  }

  /* Search through outputstrings to see if any oscillator contains "fullstate" */
  writefullstate = false;
  for (int i=0; i<outputstr.size(); i++) {
    for (int j=0; j<outputstr[i].size(); j++) {
      if (outputstr[i][j].compare("fullstate") == 0 ) writefullstate = true;
    }
  }

  /* Prepare data output files */
  ufile = NULL;
  vfile = NULL;
  for (int i=0; i< outputstr.size(); i++) expectedfile.push_back (NULL);
  for (int i=0; i< outputstr.size(); i++) populationfile.push_back (NULL);

}

Output::Output(MapParam& config, int mpirank_petsc_, int mpirank_init_, int mpirank_braid_) : Output(config, mpirank_petsc_, mpirank_init_) {
  mpirank_braid = mpirank_braid_;
}


Output::~Output(){
  printf("Output directory: %s\n", datadir.c_str());
  if (mpirank_world == 0) fclose(optimfile);
}


void Output::writeOptimFile(double objective, double gnorm, double stepsize, double cost, double tikh_regul, double penalty){

  if (mpirank_world == 0){
    fprintf(optimfile, "%05d  %1.14e  %1.14e  %.8f  %1.14e  %1.14e  %1.14e\n", optim_iter, objective, gnorm, stepsize, cost, tikh_regul, penalty);
    fflush(optimfile);
  } 

}


void Output::writeControls(Vec params, MasterEq* mastereq, int ntime, double dt){

  /* Write controls every <outfreq> iterations */
  if ( mpirank_world == 0 && optim_iter % optim_monitor_freq == 0 ) { 

    char filename[255];
    int ndesign;
    VecGetSize(params, &ndesign);

    /* Print current parameters to file */
    FILE *paramfile;
    sprintf(filename, "%s/optim_param_iter%04d.dat", datadir.c_str(), optim_iter);
    paramfile = fopen(filename, "w");

    const PetscScalar* params_ptr;
    VecGetArrayRead(params, &params_ptr);
    for (int i=0; i<ndesign; i++){
      fprintf(paramfile, "%1.14e\n", params_ptr[i]);
    }
    fclose(paramfile);
    VecRestoreArrayRead(params, &params_ptr);

    /* Print control functions */
    mastereq->setControlAmplitudes(params);
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
        sprintf(filename, "%s/control%d_iter%04d.dat", datadir.c_str(), ioscil, optim_iter);
        mastereq->getOscillator(ioscil)->flushControl(ntime, dt, filename);
    }

  }
}


void Output::openDataFiles(std::string prefix, int initid, int rank){
  char filename[255];

  bool write_this_iter = false;
  if (optim_iter % optim_monitor_freq == 0) write_this_iter = true;

  /* Open files for state vector */
  if (writefullstate && write_this_iter) {
    sprintf(filename, "%s/%s_Re.iinit%04d.rank%04d.dat", datadir.c_str(), prefix.c_str(), initid, rank);
    ufile = fopen(filename, "w");
    sprintf(filename, "%s/%s_Im.iinit%04d.rank%04d.dat", datadir.c_str(), prefix.c_str(), initid, rank);
    vfile = fopen(filename, "w"); 
  }

  /* Open files for expected energy */
  for (int i=0; i<outputstr.size(); i++) {
    for (int j=0; j<outputstr[i].size(); j++) {
      if (outputstr[i][j].compare("expectedEnergy") == 0 && write_this_iter) {
        sprintf(filename, "%s/expected%d.iinit%04d.rank%04d.dat", datadir.c_str(), i, initid, rank);
        expectedfile[i] = fopen(filename, "w");
        fprintf(expectedfile[i], "# time      expected energy level\n");
      }
      if (outputstr[i][j].compare("population") == 0 && write_this_iter) {
        sprintf(filename, "%s/population%d.iinit%04d.rank%04d.dat", datadir.c_str(), i, initid, rank);
        populationfile[i] = fopen(filename, "w");
        fprintf(populationfile[i], "# time      diagonal of the density matrix \n");
      }
    }
  }

}

void Output::writeDataFiles(double time, const Vec state, MasterEq* mastereq){

  /* Write expected energy levels to file */
  for (int iosc = 0; iosc < expectedfile.size(); iosc++) {
    if (expectedfile[iosc] != NULL) {
      double expected = mastereq->getOscillator(iosc)->expectedEnergy(state);
      fprintf(expectedfile[iosc], "%.8f %1.14e\n", time, expected);
    }
  }

  /* Write population to file */
  for (int iosc = 0; iosc < populationfile.size(); iosc++) {
    if (populationfile[iosc] != NULL) {
      std::vector<double> pop (mastereq->getOscillator(iosc)->getNLevels(), 0.0);
      mastereq->getOscillator(iosc)->population(state, pop);
      fprintf(populationfile[iosc], "%.8f ", time);
      for (int i = 0; i<pop.size(); i++) {
        fprintf(populationfile[iosc], " %1.14e", pop[i]);
      }
      fprintf(populationfile[iosc], "\n");
    }
  }



  /* Write full state to file */
  if (writefullstate) {

    /* Gather the vector from all petsc processors onto the first one */
    // VecScatterCreateToZero(x, &scat, &xseq);
    // VecScatterBegin(scat, u->x, xseq, INSERT_VALUES, SCATTER_FORWARD);
    // VecScatterEnd(scat, u->x, xseq, INSERT_VALUES, SCATTER_FORWARD);

    /* Write full state vector to file */
    if (ufile != NULL && vfile != NULL) {
      fprintf(ufile,  "%.8f  ", time);
      fprintf(vfile,  "%.8f  ", time);

      const PetscScalar *x;
      VecGetArrayRead(state, &x);
      for (int i=0; i<mastereq->getDim(); i++) {
        fprintf(ufile, "%1.10e  ", x[getIndexReal(i)]);  
        fprintf(vfile, "%1.10e  ", x[getIndexImag(i)]);  
      }
      fprintf(ufile, "\n");
      fprintf(vfile, "\n");
      VecRestoreArrayRead(state, &x);
    }
      /* Destroy scatter context and vector */
      // VecScatterDestroy(&scat);
      // VecDestroy(&xseq); // TODO create and destroy scatter and xseq in contructor/destructor
  }
}

void Output::closeDataFiles(){

  /* Close output data files */
  if (ufile != NULL) {
    fclose(ufile);
    ufile = NULL;
  }
  if (vfile != NULL) {
    fclose(vfile);
    vfile = NULL;
  }
  for (int i=0; i< expectedfile.size(); i++) {
    if (expectedfile[i] != NULL) {
      fclose(expectedfile[i]);
      expectedfile[i] = NULL;
    }
  }
  for (int i=0; i< populationfile.size(); i++) {
    if (populationfile[i] != NULL) {
      fclose(populationfile[i]);
      populationfile[i] = NULL;
    }
  }
}