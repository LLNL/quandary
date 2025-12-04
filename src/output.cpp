#include "output.hpp"
#include "defs.hpp"
#include <vector>

Output::Output(){
  mpirank_world = -1;
  mpirank_petsc = -1;
  mpirank_init  = -1;
  output_frequency = 0;
  quietmode = false;
}

Output::Output(const Config& config, MPI_Comm comm_petsc, MPI_Comm comm_init, bool quietmode_) : Output() {

  size_t noscillators = config.getNumOsc();

  /* Get communicator ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_rank(comm_petsc, &mpirank_petsc);
  MPI_Comm_size(comm_petsc, &mpisize_petsc);
  MPI_Comm_rank(comm_init, &mpirank_init);

  /* Reduced output */
  quietmode = quietmode_;


  /* Create Data directory */
  datadir = config.getDataDir();
  if (mpirank_world == 0) {
    mkdir(datadir.c_str(), 0777);
  }

  /* Prepare output for optimizer */
  optim_monitor_freq = config.getOptimMonitorFrequency();
  output_frequency = config.getOutputFrequency();
  if (mpirank_world == 0) {
    char filename[255];
    snprintf(filename, 254, "%s/optim_history.dat", datadir.c_str());
    optimfile = fopen(filename, "w");
    if (optimfile == nullptr) {
      printf("ERROR: Could not open file %s\n", filename);
    }
    fprintf(optimfile, "#\"iter\"    \"Objective\"           \"||Pr(grad)||\"           \"LS step\"           \"F_avg\"           \"Terminal cost\"         \"Tikhonov-regul\"        \"Penalty-term\"          \"State variation\"        \"Energy-term\"           \"Control variation\"\n");
  } 

  /* Reset flags and data file pointers */
  ufile = NULL;
  vfile = NULL;
  for (size_t i=0; i< noscillators; i++) expectedfile.push_back (NULL);
  for (size_t i=0; i< noscillators; i++) populationfile.push_back (NULL);
  expectedfile_comp=NULL;
  populationfile_comp=NULL;
  writeFullState = false;
  writeExpectedEnergy_comp = false;
  writePopulation_comp = false;
  for (size_t i=0; i<noscillators; i++) writeExpectedEnergy.push_back(false);
  for (size_t i=0; i<noscillators; i++) writePopulation.push_back(false);

  /* Check the output strings for each oscillator to determine which files should be written */
  output = config.getOutput();
  for (size_t i=0; i<noscillators; i++) { // iterates over oscillators
    for (auto output_type : output[i]) { // iterates over output types for this oscillator
      switch (output_type) {
        case OutputType::EXPECTED_ENERGY:
          writeExpectedEnergy[i] = true;
          break;
        case OutputType::EXPECTED_ENERGY_COMPOSITE:
          writeExpectedEnergy_comp = true;
          break;
        case OutputType::POPULATION:
          writePopulation[i] = true;
          break;
        case OutputType::POPULATION_COMPOSITE:
          writePopulation_comp = true;
          break;
        case OutputType::FULLSTATE:
          writeFullState = true;
          break;
      }
    }
  }
}


Output::~Output(){
  if (mpirank_world == 0 && !quietmode) printf("Output directory: %s\n", datadir.c_str());
  if (mpirank_world == 0) fclose(optimfile);
  writeExpectedEnergy.clear();
  writePopulation.clear();
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
    if (file == nullptr) {
      printf("ERROR: Could not open file %s\n", filename);
    }

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

void Output::writeControls(Vec params, MasterEq* mastereq, int ntime, double dt){

  /* Write controls every <outfreq> iterations */
  if ( mpirank_world == 0 ) { 

    char filename[255];
    PetscInt ndesign;
    VecGetSize(params, &ndesign);

    /* Print current parameters to file */
    FILE *file, *file_c;
    snprintf(filename, 254, "%s/params.dat", datadir.c_str());
    file = fopen(filename, "w");
    if (file == nullptr) {
      printf("ERROR: Could not open file %s\n", filename);
    }

    const PetscScalar* params_ptr;
    VecGetArrayRead(params, &params_ptr);
    for (int i=0; i<ndesign; i++){
      fprintf(file, "%1.14e\n", params_ptr[i]);
    }
    fclose(file);
    VecRestoreArrayRead(params, &params_ptr);
    if (!quietmode) printf("File written: %s\n", filename);

    /* Print control to file for each oscillator */
    mastereq->setControlAmplitudes(params);
    for (size_t ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      snprintf(filename, 254, "%s/control%zu.dat", datadir.c_str(), ioscil);
      file_c = fopen(filename, "w");
      if (file_c == nullptr) {
        printf("ERROR: Could not open file %s\n", filename);
      }
      fprintf(file_c, "#\"time\"         \"p(t) (rotating)\"          \"q(t) (rotating)\"         \"f(t) (labframe)\"\n");

      /* Write every <num> timestep to file */
      for (int i=0; i<=ntime; i+=output_frequency) {
        double time = i*dt; 

        double ReI, ImI, LabI;
        mastereq->getOscillator(ioscil)->evalControl(time, &ReI, &ImI);
        mastereq->getOscillator(ioscil)->evalControl_Labframe(time, &LabI);
        // Write control drives
        fprintf(file_c, "% 1.8f   % 1.14e   % 1.14e   % 1.14e \n", time, ReI/(2.0*M_PI), ImI/(2.0*M_PI), LabI/(2.0*M_PI));
     } // end of time loop 

      fclose(file_c);
      if (!quietmode) printf("File written: %s\n", filename);
    } // end of oscillator loop
  }
}


void Output::openTrajectoryDataFiles(std::string prefix, int initid){
  printf("DEBUG openTrajectoryDataFiles: Starting with prefix='%s', initid=%d, mpirank_petsc=%d\n", prefix.c_str(), initid, mpirank_petsc);
  char filename[255];

  // On the first petsc rank, open required files and print header information
  printf("DEBUG openTrajectoryDataFiles: About to check mpirank_petsc == 0\n");
  if (mpirank_petsc == 0) {
    printf("DEBUG openTrajectoryDataFiles: Inside mpirank_petsc == 0 block\n");

    // Expected energy per oscillator  
    for (size_t i=0; i<output.size(); i++) { // iterates over oscillators
      if (writeExpectedEnergy[i]) {
        snprintf(filename, 254, "%s/expected%zu.iinit%04d.dat", datadir.c_str(), i, initid);
        expectedfile[i] = fopen(filename, "w");
        if (expectedfile[i] == nullptr) {
          printf("ERROR: Could not open file %s\n", filename);
        }
        fprintf(expectedfile[i], "#\"time\"      \"expected energy level\"\n");
      }
    }
    // Expected energy for full composite system
    if (writeExpectedEnergy_comp) {
      snprintf(filename, 254, "%s/expected_composite.iinit%04d.dat", datadir.c_str(), initid);
      expectedfile_comp = fopen(filename, "w");
      if (expectedfile_comp == nullptr) {
        printf("ERROR: Could not open file %s\n", filename);
      }
      fprintf(expectedfile_comp, "#\"time\"      \"expected energy level\"\n");
    }
    // Populations per oscillator
    for (size_t i=0; i<output.size(); i++) { // iterates over oscillators
      if (writePopulation[i]) {
        snprintf(filename, 254, "%s/population%zu.iinit%04d.dat", datadir.c_str(), i, initid);
        populationfile[i] = fopen(filename, "w");
        if (populationfile[i] == nullptr) {
          printf("ERROR: Could not open file %s\n", filename);
        }
        fprintf(populationfile[i], "#\"time\"      \"diagonal of the density matrix\"\n");
      }
    }
    // Population for full composite system 
    if (writePopulation_comp) {
      snprintf(filename, 254, "%s/population_composite.iinit%04d.dat", datadir.c_str(), initid);
      populationfile_comp = fopen(filename, "w");
      if (populationfile_comp == nullptr) {
        printf("ERROR: Could not open file %s\n", filename);
      }
      fprintf(populationfile_comp, "#\"time\"      \"population\"\n");
    }
    // Full vectorized state 
    if (writeFullState ) {
      snprintf(filename, 254, "%s/%s_Re.iinit%04d.dat", datadir.c_str(), prefix.c_str(), initid);
      ufile = fopen(filename, "w");
      if (ufile == nullptr) {
        printf("ERROR: Could not open file %s\n", filename);
      }
      snprintf(filename, 254, "%s/%s_Im.iinit%04d.dat", datadir.c_str(), prefix.c_str(), initid);
      vfile = fopen(filename, "w"); 
      if (vfile == nullptr) {
        printf("ERROR: Could not open file %s\n", filename);
      }
    }
  }
}

void Output::writeTrajectoryDataFiles(int timestep, double time, const Vec state, MasterEq* mastereq){

  /* Write output only every <num> time-steps */
  if (timestep % output_frequency == 0) {
    if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Starting timestep=0\n", mpirank_world, mpirank_petsc);

    /* Write expected energy levels to file */
    if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to loop through expectedfile.size()=%zu\n", mpirank_world, mpirank_petsc, expectedfile.size());
    for (size_t iosc = 0; iosc < expectedfile.size(); iosc++) {
      if (writeExpectedEnergy[iosc]) {
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to call expectedEnergy for iosc=%zu\n", mpirank_world, mpirank_petsc, iosc);
        double expected = mastereq->getOscillator(iosc)->expectedEnergy(state);
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed expectedEnergy for iosc=%zu, expected=%f\n", mpirank_world, mpirank_petsc, iosc, expected);
        if (mpirank_petsc==0) fprintf(expectedfile[iosc], "%.8f %1.14e\n", time, expected);
      }
    }
    if (writeExpectedEnergy_comp) {
      if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to call mastereq->expectedEnergy\n", mpirank_world, mpirank_petsc);
      double expected_comp = mastereq->expectedEnergy(state);
      if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed mastereq->expectedEnergy, expected_comp=%f\n", mpirank_world, mpirank_petsc, expected_comp);
      if (mpirank_petsc==0) fprintf(expectedfile_comp, "%.8f %1.14e\n", time, expected_comp);
    }

    /* Write population to file */
    if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to loop through populationfile.size()=%zu\n", mpirank_world, mpirank_petsc, populationfile.size());
    for (size_t iosc = 0; iosc < populationfile.size(); iosc++) {
      if (writePopulation[iosc]) {
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to create pop vector for iosc=%zu\n", mpirank_world, mpirank_petsc, iosc);
        std::vector<double> pop (mastereq->getOscillator(iosc)->getNLevels(), 0.0);
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to call population for iosc=%zu\n", mpirank_world, mpirank_petsc, iosc);
        mastereq->getOscillator(iosc)->population(state, pop);
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed population for iosc=%zu\n", mpirank_world, mpirank_petsc, iosc);
        if (mpirank_petsc == 0) {
          fprintf(populationfile[iosc], "%.8f ", time);
          for (size_t i = 0; i<pop.size(); i++) {
            fprintf(populationfile[iosc], " %1.14e", pop[i]);
          }
          fprintf(populationfile[iosc], "\n");
        }
      }
    }
    if (writePopulation_comp) {
      if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to call mastereq->population composite\n", mpirank_world, mpirank_petsc);
      std::vector<double> population_comp; 
      mastereq->population(state, population_comp);
      if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed mastereq->population composite\n", mpirank_world, mpirank_petsc);
      if (mpirank_petsc == 0) {
        fprintf(populationfile_comp, "%.8f  ", time);
        for (size_t i=0; i<population_comp.size(); i++){
          fprintf(populationfile_comp, "%1.14e  ", population_comp[i]);
        }
        fprintf(populationfile_comp, "\n");
      }
    }

    /* Write full state to file. Currently not available if Petsc-parallel */
    if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to check writeFullState=%s, mpisize_petsc=%d\n", mpirank_world, mpirank_petsc, writeFullState ? "true" : "false", mpisize_petsc);
    if (writeFullState && mpisize_petsc == 1) {
      /* TODO: Make this work in parallel! */
      /* Gather the vector from all petsc processors onto the first one */
      // VecScatterCreateToZero(x, &scat, &xseq);
      // VecScatterBegin(scat, u->x, xseq, INSERT_VALUES, SCATTER_FORWARD);
      // VecScatterEnd(scat, u->x, xseq, INSERT_VALUES, SCATTER_FORWARD);

      /* On first petsc rank, write full state vector to file */
      if (mpirank_petsc == 0) {
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to write full state\n", mpirank_world, mpirank_petsc);
        fprintf(ufile,  "%.8f  ", time);
        fprintf(vfile,  "%.8f  ", time);
        const PetscScalar *x;
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to call VecGetArrayRead\n", mpirank_world, mpirank_petsc);
        VecGetArrayRead(state, &x);
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed VecGetArrayRead\n", mpirank_world, mpirank_petsc);
        for (int i=0; i<mastereq->getDim(); i++) {
          fprintf(ufile, "%1.10e  ", x[i]);  
          fprintf(vfile, "%1.10e  ", x[i + mastereq->getDim()]);  
        }
        fprintf(ufile, "\n");
        fprintf(vfile, "\n");
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: About to call VecRestoreArrayRead\n", mpirank_world, mpirank_petsc);
        VecRestoreArrayRead(state, &x);
        if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed VecRestoreArrayRead\n", mpirank_world, mpirank_petsc);
      }
      /* Destroy scatter context and vector */
      // VecScatterDestroy(&scat);
      // VecDestroy(&xseq); // TODO create and destroy scatter and xseq in contructor/destructor
    }
    if (timestep == 0) printf("[%d,%d] DEBUG writeTrajectoryDataFiles: Completed function for timestep=0\n", mpirank_world, mpirank_petsc);
  }
}

void Output::closeTrajectoryDataFiles(){

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
}
