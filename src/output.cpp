#include "output.hpp"

Output::Output(){
  mpirank_world = -1;
  mpirank_petsc = -1;
  mpirank_init  = -1;
  mpirank_braid = -1;
  optim_monitor_freq = 0;
  output_frequency = 0;
  optim_iter = 0;
}

Output::Output(MapParam& config, MPI_Comm comm_petsc, MPI_Comm comm_init, int noscillators) : Output() {

  /* Get communicator ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_rank(comm_petsc, &mpirank_petsc);
  MPI_Comm_size(comm_petsc, &mpisize_petsc);
  MPI_Comm_rank(comm_init, &mpirank_init);


  /* Create Data directory */
  datadir = config.GetStrParam("datadir", "./data_out");
  if (mpirank_world == 0) {
    mkdir(datadir.c_str(), 0777);
  }

  /* Prepare output for optimizer */
  optim_monitor_freq = config.GetIntParam("optim_monitor_frequency", 10);
  output_frequency = config.GetIntParam("output_frequency", 1);
  if (mpirank_world == 0) {
    char filename[255];
    sprintf(filename, "%s/optim_history.dat", datadir.c_str());
    optimfile = fopen(filename, "w");
    fprintf(optimfile, "#iter    Objective           ||Pr(grad)||           LS step           F_avg           Terminal cost       Tikhonov-regul      Penalty-term\n");
  } 

  /* Read from config what output is desired */
  for (int i = 0; i < noscillators; i++){
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

Output::Output(MapParam& config, MPI_Comm comm_petsc, MPI_Comm comm_init, MPI_Comm comm_braid, int noscillators) : Output(config, comm_petsc, comm_init, noscillators) {
  MPI_Comm_rank(comm_braid, &mpirank_braid);
}


Output::~Output(){
  if (mpirank_world == 0) printf("Output directory: %s\n", datadir.c_str());
  if (mpirank_world == 0) fclose(optimfile);
}


void Output::writeOptimFile(double objective, double gnorm, double stepsize, double Favg, double costT, double tikh_regul, double penalty){

  if (mpirank_world == 0){
    fprintf(optimfile, "%05d  %1.14e  %1.14e  %.8f  %1.14e  %1.14e  %1.14e  %1.14e\n", optim_iter, objective, gnorm, stepsize, Favg, costT, tikh_regul, penalty);
    fflush(optimfile);
  } 

}

void Output::writeGradient(Vec grad){
  char filename[255];  
  PetscInt ngrad;
  VecGetSize(grad, &ngrad);

  if (mpirank_world == 0) {
    /* Print current gradients to file */
    FILE *file;
    // sprintf(filename, "%s/grad_iter%04d.dat", datadir.c_str(), optim_iter);
    sprintf(filename, "%s/grad.dat", datadir.c_str());
    file = fopen(filename, "w");

    const PetscScalar* grad_ptr;
    VecGetArrayRead(grad, &grad_ptr);
    for (int i=0; i<ngrad; i++){
      fprintf(file, "%1.14e\n", grad_ptr[i]);
    }
    fclose(file);
    VecRestoreArrayRead(grad, &grad_ptr);
    printf("File written: %s\n", filename);
  }
}

void Output::writeControls(Vec params, MasterEq* mastereq, int ntime, double dt){

  /* Write controls every <outfreq> iterations */
  if ( mpirank_world == 0 && optim_iter % optim_monitor_freq == 0 ) { 

    char filename[255];
    PetscInt ndesign;
    VecGetSize(params, &ndesign);

    /* Print current parameters to file */
    FILE *file;
    // sprintf(filename, "%s/params_iter%04d.dat", datadir.c_str(), optim_iter);
    sprintf(filename, "%s/params.dat", datadir.c_str());
    file = fopen(filename, "w");

    const PetscScalar* params_ptr;
    VecGetArrayRead(params, &params_ptr);
    for (int i=0; i<ndesign; i++){
      fprintf(file, "%1.14e\n", params_ptr[i]);
    }
    fclose(file);
    VecRestoreArrayRead(params, &params_ptr);
    printf("File written: %s\n", filename);

    /* Print control functions to file */
    mastereq->setControlAmplitudes(params);
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      sprintf(filename, "%s/control%d.dat", datadir.c_str(), ioscil);
      file = fopen(filename, "w");
      fprintf(file, "# time         p(t) (rotating)          q(t) (rotating)        f(t) (labframe) \n");

      /* Write every <num> timestep to file */
      for (int i=0; i<=ntime; i+=output_frequency) {
        double time = i*dt; 
        double Re, Im, Lab;
        mastereq->getOscillator(ioscil)->evalControl(time, &Re, &Im);
        mastereq->getOscillator(ioscil)->evalControl_Labframe(time, &Lab);
        fprintf(file, "% 1.8f   % 1.14e   % 1.14e   % 1.14e \n", time, Re, Im, Lab);
      }

      fclose(file);
      printf("File written: %s\n", filename);
    }

  }
}


void Output::openDataFiles(std::string prefix, int initid){
  char filename[255];

  /* Flag to determine of this optimization iteration will write data output */
  bool write_this_iter = false;
  if (optim_iter % optim_monitor_freq == 0) write_this_iter = true;

  /* Check if Xbraid is running in parallel and prepare output names if it is.*/
  char postchar[255];
  sprintf(postchar,"");
  if (mpirank_braid >= 0) sprintf(postchar, ".braidrank%04d", mpirank_braid);

  /* Open files for state vector */
  if (mpirank_petsc == 0 && writefullstate && write_this_iter) {
    sprintf(filename, "%s/%s_Re.iinit%04d%s.dat", datadir.c_str(), prefix.c_str(), initid, postchar);
    ufile = fopen(filename, "w");
    sprintf(filename, "%s/%s_Im.iinit%04d%s.dat", datadir.c_str(), prefix.c_str(), initid, postchar);
    vfile = fopen(filename, "w"); 
  }

  /* Open files for expected energy */
  if (mpirank_petsc == 0 && write_this_iter) {
    for (int i=0; i<outputstr.size(); i++) {
      for (int j=0; j<outputstr[i].size(); j++) {
        if (outputstr[i][j].compare("expectedEnergy") == 0) {
          sprintf(filename, "%s/expected%d.iinit%04d%s.dat", datadir.c_str(), i, initid, postchar);
          expectedfile[i] = fopen(filename, "w");
          fprintf(expectedfile[i], "# time      expected energy level\n");
        }
        if (outputstr[i][j].compare("population") == 0) {
          sprintf(filename, "%s/population%d.iinit%04d%s.dat", datadir.c_str(), i, initid, postchar);
          populationfile[i] = fopen(filename, "w");
          fprintf(populationfile[i], "# time      diagonal of the density matrix \n");
        }
      }
    }
  }

}

void Output::writeDataFiles(int timestep, double time, const Vec state, MasterEq* mastereq){

  /* Write output only every <num> time-steps */
  if (timestep % output_frequency == 0) {

    /* Write expected energy levels to file */
    for (int iosc = 0; iosc < expectedfile.size(); iosc++) {
      double expected = mastereq->getOscillator(iosc)->expectedEnergy(state);
      if (expectedfile[iosc] != NULL) fprintf(expectedfile[iosc], "%.8f %1.14e\n", time, expected);
    }

    /* Write population to file */
    for (int iosc = 0; iosc < populationfile.size(); iosc++) {
      std::vector<double> pop (mastereq->getOscillator(iosc)->getNLevels(), 0.0);
      mastereq->getOscillator(iosc)->population(state, pop);
      // write
      if (populationfile[iosc] != NULL) {
        fprintf(populationfile[iosc], "%.8f ", time);
        for (int i = 0; i<pop.size(); i++) {
          fprintf(populationfile[iosc], " %1.14e", pop[i]);
        }
        fprintf(populationfile[iosc], "\n");
      }
    }

    /* Write full state to file */
    if (writefullstate && mpisize_petsc == 1) {

      /* TODO: Make this work in parallel! */
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
