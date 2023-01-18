#include "output.hpp"

Output::Output(){
  mpirank_world = 0;
  mpirank_petsc = 0;
  mpirank_init  = 0;
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
  expectedfile_comp=NULL;
  populationfile_comp=NULL;

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
    char filename_transfer[255];
    PetscInt ndesign;
    VecGetSize(params, &ndesign);

    /* Print current parameters to file */
    FILE *file, *file_c, *file_t;
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

    /* Print control p(t) and transfer u_i(p(t)) to file for each oscillator */
    mastereq->setControlAmplitudes(params);
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      sprintf(filename, "%s/control%d.dat", datadir.c_str(), ioscil);
      sprintf(filename_transfer, "%s/transfer_control%d.dat", datadir.c_str(), ioscil);
      file_c = fopen(filename, "w");
      file_t = fopen(filename_transfer, "w");
      fprintf(file_c, "# time         p(t) (rotating)          q(t) (rotating)        f(t) (labframe) \n");
      fprintf(file_t, "# time     u^k_1(p(t))     u^k_2(p(t))    ...      v^k_1(q(t))     v^k_2(q(t))     ...\n");

      /* Write every <num> timestep to file */
      for (int i=0; i<=ntime; i+=output_frequency) {
        double time = i*dt; 

        double ReI, ImI, LabI;
        mastereq->getOscillator(ioscil)->evalControl(time, &ReI, &ImI);
        mastereq->getOscillator(ioscil)->evalControl_Labframe(time, &LabI);
        // Write control drives
        fprintf(file_c, "% 1.8f   % 1.14e   % 1.14e   % 1.14e \n", time, ReI, ImI, LabI);
        
        // Evaluate and write transfer functions u_i(p(t)), v_i(q(t)) for this oscillator
        fprintf(file_t, "% 1.8f   ", time);
        for (int icon=0; icon<mastereq->transfer_Hc_re[ioscil].size(); icon++){
          double ukip = mastereq->transfer_Hc_re[ioscil][icon]->eval(ReI, time);
          fprintf(file_t, "% 1.14e   ", ukip);
        }
        // Get transfer functions v^k_i(q) for this oscillators k
        for (int icon=0; icon<mastereq->transfer_Hc_im[ioscil].size(); icon++){
          double ukiq = mastereq->transfer_Hc_im[ioscil][icon]->eval(ImI, time);
          fprintf(file_t, "% 1.14e   ", ukiq);
        } 
        fprintf(file_t, "\n");
      } // end of time loop 

      fclose(file_c);
      fclose(file_t);
      printf("File written: %s\n", filename);
      printf("File written: %s\n", filename_transfer);
    } // end of oscillator loop
  }
}


void Output::openDataFiles(std::string prefix, int initid){
  char filename[255];

  /* Flag to determine of this optimization iteration will write data output */
  bool write_this_iter = false;
  if (optim_iter % optim_monitor_freq == 0) write_this_iter = true;

  /* Open files for state vector */
  if (mpirank_petsc == 0 && writefullstate && write_this_iter) {
    sprintf(filename, "%s/%s_Re.iinit%04d.dat", datadir.c_str(), prefix.c_str(), initid);
    ufile = fopen(filename, "w");
    sprintf(filename, "%s/%s_Im.iinit%04d.dat", datadir.c_str(), prefix.c_str(), initid);
    vfile = fopen(filename, "w"); 
  }

  /* Open files for expected energy */
  bool writeExpComp = false;
  bool writePopComp = false;
  if (mpirank_petsc == 0 && write_this_iter) {
    for (int i=0; i<outputstr.size(); i++) {
      for (int j=0; j<outputstr[i].size(); j++) {
        if (outputstr[i][j].compare("expectedEnergy") == 0) {
          sprintf(filename, "%s/expected%d.iinit%04d.dat", datadir.c_str(), i, initid);
          expectedfile[i] = fopen(filename, "w");
          fprintf(expectedfile[i], "# time      expected energy level\n");
        }
        if (outputstr[i][j].compare("expectedEnergyComposite") == 0) writeExpComp = true;
        if (outputstr[i][j].compare("population") == 0) {
          sprintf(filename, "%s/population%d.iinit%04d.dat", datadir.c_str(), i, initid);
          populationfile[i] = fopen(filename, "w");
          fprintf(populationfile[i], "# time      diagonal of the density matrix \n");
        }
        if (outputstr[i][j].compare("populationComposite") == 0) writePopComp = true;
      }
    }
    if (writeExpComp){
      sprintf(filename, "%s/expected_composite.iinit%04d.dat", datadir.c_str(), initid);
      expectedfile_comp = fopen(filename, "w");
      fprintf(expectedfile_comp, "# time      expected energy level\n");
    }
    if (writePopComp){
      sprintf(filename, "%s/population_composite.iinit%04d.dat", datadir.c_str(), initid);
      populationfile_comp = fopen(filename, "w");
      fprintf(populationfile_comp, "# time      population \n");
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

    double expected_comp = mastereq->expectedEnergy(state);
    if (expectedfile_comp != NULL) fprintf(expectedfile_comp, "%.8f %1.14e\n", time, expected_comp);

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


    std::vector<double> population_comp; 
    mastereq->population(state, population_comp);
    if (populationfile_comp != NULL) {
      fprintf(populationfile_comp, "%.8f  ", time);
      for (int i=0; i<population_comp.size(); i++){
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
  if (expectedfile_comp != NULL) fclose(expectedfile_comp);
  expectedfile_comp = NULL;
  for (int i=0; i< populationfile.size(); i++) {
    if (populationfile[i] != NULL) {
      fclose(populationfile[i]);
      populationfile[i] = NULL;
    }
  }
  if (populationfile_comp != NULL) fclose(populationfile_comp);
  populationfile_comp = NULL;
}
