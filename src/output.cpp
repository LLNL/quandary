#include "output.hpp"

Output::Output(){
  mpirank_world = -1;
  mpirank_petsc = -1;
  mpirank_init  = -1;
  output_frequency = 0;
  quietmode = false;

}

Output::Output(MapParam config, MPI_Comm comm_petsc, MPI_Comm comm_init, int noscillators, bool quietmode_) : Output() {

  /* Get communicator ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_rank(comm_petsc, &mpirank_petsc);
  MPI_Comm_size(comm_petsc, &mpisize_petsc);
  MPI_Comm_rank(comm_init, &mpirank_init);

  /* Reduced output */
  quietmode = quietmode_;


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
    snprintf(filename, 254, "%s/optim_history.dat", datadir.c_str());
    optimfile = fopen(filename, "w");
    fprintf(optimfile, "#\"iter\"    \"Objective\"           \"||Pr(grad)||\"           \"LS step\"           \"F_avg\"           \"Terminal cost\"         \"Tikhonov-regul\"        \"Penalty-term\"          \"State variation\"        \"Energy-term\"           \"Control variation\"\n");
  } 

  /* Read from config what output is desired */
  for (int i = 0; i < noscillators; i++){
    std::vector<std::string> fillme;
    config.GetVecStrParam("output" + std::to_string(i), fillme, "none");
    outputstr.push_back(fillme);
  }

  /* Search through outputstrings to see if any oscillator contains "fullstate" */
  writefullstate = false;
  for (size_t i=0; i<outputstr.size(); i++) {
    for (size_t j=0; j<outputstr[i].size(); j++) {
      if (outputstr[i][j].compare("fullstate") == 0 ) writefullstate = true;
    }
  }

  /* Prepare data output files */
  ufile = NULL;
  vfile = NULL;
  for (size_t i=0; i< outputstr.size(); i++) expectedfile.push_back (NULL);
  for (size_t i=0; i< outputstr.size(); i++) populationfile.push_back (NULL);
  expectedfile_comp=NULL;
  populationfile_comp=NULL;
  errorfile = NULL;
}


Output::~Output(){
  if (mpirank_world == 0 && !quietmode) printf("Output directory: %s\n", datadir.c_str());
  if (mpirank_world == 0) fclose(optimfile);
}


void Output::writeOptimFile(int optim_iter, double objective, double gnorm, double stepsize, double Favg, double costT, double tikh_regul, double penalty, double penalty_dpdm, double penalty_energy, double penalty_variation){

  if (mpirank_world == 0){
    fprintf(optimfile, "%05d  %1.14e  %1.14e  %.8f  %1.14e  %1.14e  %1.14e  %1.14e  %1.14e  %1.14e  %1.14e\n", optim_iter, objective, gnorm, stepsize, Favg, costT, tikh_regul, penalty, penalty_dpdm, penalty_energy, penalty_variation);
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
    snprintf(filename, 254, "%s/grad.dat", datadir.c_str());
    file = fopen(filename, "w");

    const PetscScalar* grad_ptr;
    VecGetArrayRead(grad, &grad_ptr);
    for (int i=0; i<ngrad; i++){
      fprintf(file, "%1.14e\n", grad_ptr[i]);
    }
    fclose(file);
    VecRestoreArrayRead(grad, &grad_ptr);
    // if (!quietmode) printf("File written: %s\n", filename);
  }
}

void Output::writeParams(Vec params) {

  if ( mpirank_world == 0 ) { 
    FILE *file;
    char filename[255];
    PetscInt ndesign;
    VecGetSize(params, &ndesign);
    snprintf(filename, 254, "%s/params.dat", datadir.c_str());
    file = fopen(filename, "w");

    const PetscScalar* params_ptr;
    VecGetArrayRead(params, &params_ptr);
    for (int i=0; i<ndesign; i++){
      fprintf(file, "%1.14e\n", params_ptr[i]);
    }
    fclose(file);
    VecRestoreArrayRead(params, &params_ptr);
    // if (!quietmode) printf("File written: %s\n", filename);
  }
}

void Output::writeControls(MasterEq* mastereq, int ntime, double dt, int pulseID){

  /* Write controls every <outfreq> iterations */
  // if ( mpirank_world == 0 ) { 

    FILE *file_c;
    char filename[255];

    /* Print control pulse for each oscillator to file */
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      if (pulseID >= 0){
        snprintf(filename, 254, "%s/control%d_pulse%d.dat", datadir.c_str(), ioscil, pulseID);
      } else {
        snprintf(filename, 254, "%s/control%d.dat", datadir.c_str(), ioscil);
      }
      file_c = fopen(filename, "w");
      fprintf(file_c, "#\"time\"         \"p(t) (rotating)\"          \"q(t) (rotating)\"         \"f(t) (labframe)\"\n");

      /* Write every <num> timestep to file */
      for (int i=0; i<=ntime; i+=output_frequency) {
        double time = i*dt; 

        double ReI, ImI, LabI;
        mastereq->getOscillator(ioscil)->evalControl(time, &ReI, &ImI, mastereq->learning);
        mastereq->getOscillator(ioscil)->evalControl_Labframe(time, &LabI, mastereq->learning);
        // Write control drives
        fprintf(file_c, "% 1.8f   % 1.14e   % 1.14e   % 1.14e \n", time, ReI/(2.0*M_PI), ImI/(2.0*M_PI), LabI/(2.0*M_PI));
     } // end of time loop 

      fclose(file_c);
      // if (!quietmode) printf("File written: %s\n", filename);
    } // end of oscillator loop
  // }
}


void Output::openDataFiles(std::string prefix, int initid, int pulseID){
  char filenameu[255];
  char filenamev[255];

  /* Flag to determine if this optimization iteration will write data output */

  /* Open files for state vector */
  if (mpirank_petsc == 0 && writefullstate ) {
    if (pulseID >= 0) {
      snprintf(filenameu, 254, "%s/%s_Re_pulse%d.iinit%04d.dat", datadir.c_str(), prefix.c_str(), pulseID, initid);
      snprintf(filenamev, 254, "%s/%s_Im_pulse%d.iinit%04d.dat", datadir.c_str(), prefix.c_str(), pulseID, initid);
    } else {
      snprintf(filenameu, 254, "%s/%s_Re.iinit%04d.dat", datadir.c_str(), prefix.c_str(), initid);
      snprintf(filenamev, 254, "%s/%s_Im.iinit%04d.dat", datadir.c_str(), prefix.c_str(), initid);
    }
    ufile = fopen(filenameu, "w");
    vfile = fopen(filenamev, "w"); 
  }

  /* Open files for expected energy */
  char filename[255];
  bool writeExpComp = false;
  bool writePopComp = false;
  if (mpirank_petsc == 0) {
    for (size_t i=0; i<outputstr.size(); i++) {
      for (size_t j=0; j<outputstr[i].size(); j++) {
        if (outputstr[i][j].compare("expectedEnergy") == 0) {
          if (pulseID >= 0) {
            snprintf(filename, 254, "%s/expected%d_pulse%d.iinit%04d.dat", datadir.c_str(), i, pulseID, initid);
          } else {
            snprintf(filename, 254, "%s/expected%d.iinit%04d.dat", datadir.c_str(), i, initid);
          }
          expectedfile[i] = fopen(filename, "w");
          fprintf(expectedfile[i], "#\"time\"      \"expected energy level\"\n");
        }
        if (outputstr[i][j].compare("expectedEnergyComposite") == 0) writeExpComp = true;
        if (outputstr[i][j].compare("population") == 0) {
          if (pulseID >= 0) {
            snprintf(filename, 254, "%s/population%d_pulse%d.iinit%04d.dat", datadir.c_str(), i, pulseID, initid);
          } else {
            snprintf(filename, 254, "%s/population%d.iinit%04d.dat", datadir.c_str(), i, initid);
          }
          populationfile[i] = fopen(filename, "w");
          fprintf(populationfile[i], "#\"time\"      \"diagonal of the density matrix\"\n");
        }
        if (outputstr[i][j].compare("populationComposite") == 0) writePopComp = true;
      }
    }
    if (writeExpComp){
      if (pulseID >= 0){
        snprintf(filename, 254, "%s/expected_composite_pulse%d.iinit%04d.dat", datadir.c_str(), pulseID, initid);
      } else {
        snprintf(filename, 254, "%s/expected_composite.iinit%04d.dat", datadir.c_str(), initid);
      }
      expectedfile_comp = fopen(filename, "w");
      fprintf(expectedfile_comp, "#\"time\"      \"expected energy level\"\n");
    }
    if (writePopComp){
      if (pulseID >= 0){
        snprintf(filename, 254, "%s/population_composite_pulse%d.iinit%04d.dat", datadir.c_str(), pulseID, initid);
      } else{
        snprintf(filename, 254, "%s/population_composite.iinit%04d.dat", datadir.c_str(), initid);
      }
      populationfile_comp = fopen(filename, "w");
      fprintf(populationfile_comp, "#\"time\"      \"population\"\n");
    }

    if (!x_is_control){
      if (pulseID >= 0) {
        snprintf(filename, 254, "%s/TrajectoryError_pulse%d.iinit%04d.dat", datadir.c_str(), pulseID, initid);
      } else {
        snprintf(filename, 254, "%s/TrajectoryError.iinit%04d.dat", datadir.c_str(), initid);
      }
      errorfile = fopen(filename, "w");
      fprintf(errorfile, "# time      error norm\n");
    }

  }

}

void Output::writeDataFiles(int timestep, double time, const Vec state, MasterEq* mastereq){

  /* Write output only every <num> time-steps */
  if (timestep % output_frequency == 0) {


    /* Write expected energy levels to file */
    for (size_t iosc = 0; iosc < expectedfile.size(); iosc++) {
      if (expectedfile[iosc] != NULL) {
        double expected = expectedEnergy(state, mastereq->lindbladtype, mastereq->nlevels, iosc);
        fprintf(expectedfile[iosc], "%.8f %1.14e\n", time, expected);
      }
    }
    if (expectedfile_comp != NULL) {
      double expected_comp = expectedEnergy(state, mastereq->lindbladtype, mastereq->nlevels);
      fprintf(expectedfile_comp, "%.8f %1.14e\n", time, expected_comp);
    }

    /* Write population to file */
    for (size_t iosc = 0; iosc < populationfile.size(); iosc++) {
      std::vector<double> pop (mastereq->getOscillator(iosc)->getNLevels(), 0.0);
      if (populationfile[iosc] != NULL) {
        fprintf(populationfile[iosc], "%.8f ", time);
        for (size_t i = 0; i<pop.size(); i++) {
          fprintf(populationfile[iosc], " %1.14e", pop[i]);
        }
        fprintf(populationfile[iosc], "\n");
      }
    }

    std::vector<double> population_comp; 
    if (populationfile_comp != NULL) {
      std::vector<double> population_comp; 
      population(state, mastereq->lindbladtype, population_comp);
      fprintf(populationfile_comp, "%.8f  ", time);
      for (size_t i=0; i<population_comp.size(); i++){
        fprintf(populationfile_comp, "%1.14e  ", population_comp[i]);
      }
      fprintf(populationfile_comp, "\n");
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

void Output::writeErrorFile(double time, double errnorm){
  if (errorfile != NULL) fprintf(errorfile, "%.8f %1.14e\n", time, errnorm);
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
  for (size_t i=0; i< expectedfile.size(); i++) {
    if (expectedfile[i] != NULL) {
      fclose(expectedfile[i]);
      expectedfile[i] = NULL;
    }
  }
  if (expectedfile_comp != NULL) fclose(expectedfile_comp);
  expectedfile_comp = NULL;
  for (size_t i=0; i< populationfile.size(); i++) {
    if (populationfile[i] != NULL) {
      fclose(populationfile[i]);
      populationfile[i] = NULL;
    }
  }
  if (populationfile_comp != NULL) fclose(populationfile_comp);
  populationfile_comp = NULL;

  if (errorfile != NULL) fclose(errorfile);
  errorfile = NULL;
}
