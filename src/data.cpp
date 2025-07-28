#include <data.hpp>

Data::Data() {
  dim = -1;
  dim_hs = -1;
  npulses = 1;
  npulses_local = 1;
  ninit = 1;
  ninit_local = 1;
  tstart = 0.0;
  tstop = 0.0;
	dt = 0.0;
  densityData = true;
}

Data::Data(Config config, MPI_Comm comm_optim_, MPI_Comm comm_init_, int ninit_, std::vector<int> nlevels_, LindbladType lindbladtype_) {
  nlevels = nlevels_; // nlevels[i] holds the total number of energy levels in subsystem 'i'
  comm_optim = comm_optim_;
  comm_init = comm_init_;
  lindbladtype = lindbladtype_;
  ninit = ninit_;

  // Compute dimension of the full vectorized system
  dim = 1;
  for (size_t i=0; i<nlevels.size(); i++){
    dim *= nlevels[i];
  }
  dim_hs = dim;   // Dimension of the Hilbert space (N)

  if (lindbladtype != LindbladType::NONE){
      dim = dim*dim;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  MPI_Comm_rank(comm_optim, &mpirank_optim);
  MPI_Comm_size(comm_optim, &mpisize_optim);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);
  ninit_local = ninit / mpisize_init;

  /* Get some data configuration */
  tstop = config.GetDoubleParam("data_tstop", 1e+14, false);
  npulses = config.GetIntParam("data_npulses", 1, true, true);
  if (npulses %  mpisize_optim != 0) { 
    printf("ERROR: Can't distribute %d pulses over %d cores\n", npulses, mpisize_optim);
    exit(1);
  }
  npulses_local = int(npulses / mpisize_optim); 

  // Set first two outer dimensions of the data to number of local control pulses and  initial conditions. Inner dimension will be filled in the subclasses.
  data.resize(npulses_local); 
  for (int i=0; i<npulses_local; i++) {
    data[i].resize(ninit_local);
  }
  controlparams.resize(npulses_local);

  // Parse training data filenames.
  std::string data_identifier;
  std::vector<std::vector<std::string>> data_names; 
  for (int ipulse_global = 0; ipulse_global< npulses; ipulse_global++){
    // Parse config "data_name" or "data_name<i>="
    std::string key = "data_name";
    if (ipulse_global > 0) key += std::to_string(ipulse_global);
    std::vector<std::string> datastr;
    config.GetVecStrParam(key, datastr, "none, none", true, false);
    // extract data identifier and directory from the first two elements 
    data_identifier = datastr[0];
    std::string directory = datastr[1];
    // Erase the first two elements. The remaining ones hold filenames.
    datastr.erase(datastr.begin());
    datastr.erase(datastr.begin());
    // Prepend the filenames with the data direcory
    for (size_t ifile=0; ifile<datastr.size(); ifile++){
      datastr[ifile] = directory + "/" + datastr[ifile];
    }    
    // Add to vector of data names
    data_names.push_back(datastr);
  }

  // Load the data
  if (data_identifier.compare("synthetic")==0) {
    loadData_SyntheticQuandaryRho(data_names);
  } else if (data_identifier.compare("syntheticPop") == 0) {
    loadData_SyntheticQuandaryPop(data_names);
  // else if (data_identifier.compare("Tant2level")==0) loadData_Tant2Level();
  // else if (data_identifier.compare("Tant3level")==0) loadData_Tant3Level();
  } else {
    printf("Wrong data identifier: Needs prefix 'synthetic' (for synthetic density matrix data), or 'syntheticPop' (for synthetic population data), or 'Tant2level' (for Tant 2level data), or 'Tant3level' (for Tant 3level data).\n");
    exit(1);
  }
}

Data::~Data() {
  for (size_t ipulse=0; ipulse<data.size(); ipulse++){
    for (size_t iinit=0; iinit<data[ipulse].size(); iinit++){
      for (size_t itime=0; itime<data[ipulse][iinit].size(); itime++){
        VecDestroy(&(data[ipulse][iinit][itime]));
      }
      data[ipulse][iinit].clear();
    }
    data[ipulse].clear();
  }
  data.clear();
  for (size_t i=0; i<controlparams.size(); i++) {
    controlparams[i].clear();
  }
  controlparams.clear();
}

double Data::suggestTimeStepSize(double dt_old){

  int loss_every_k;
  if (abs(std::remainder(dt, dt_old)) < 1e-8) // dt_old already is an integer divisor of data_dt
    loss_every_k = std::round(dt / dt_old); 
  else // dt_old is not integer divisor of data_dt
    loss_every_k = ceil(dt / dt_old);   // Next larger integer
  
  // Updated timestep size
  double dt_new = dt / loss_every_k;    

  return dt_new;
}

Vec Data::getData(double time, int ipulse_global, int iinit_global){
  // Get local pulse number for this processor;
  int ipulse_local = ipulse_global % npulses_local;
  int iinit_local = iinit_global % ninit_local;
  
  if (tstart <= time && time <= tstop) {  // if time is within the data domain
    double remainder = std::remainder(time - tstart, dt);
    if (abs(remainder) < 1e-10) {        // if data exists at this time
      int itime = round((time - tstart)/dt); // index in time
      return data[ipulse_local][iinit_local][itime]; 
    } else return NULL;
  } else return NULL;
}

// TODO: Multiple oscillators!
std::vector<double> Data::getControls(int ipulse_global){

  // Get local pulse number for this processor;
  int ipulse_local =  ipulse_global % npulses_local;

  if (controlparams.size()>ipulse_local) {
    return controlparams[ipulse_local];
  }
  else {
    std::vector<double> dummy;
    return dummy;
  }
}


void Data::writeExpectedEnergy(const char* filename, int ipulse_global, int iinit_global, int ioscillator){
  // NOTE: only possible to evaluate the expectedEnergy if the data holds density matrix data
  if (densityData){
    // Only the processor who owns this pulse trajectory should be writing the file, other procs exit here.
    int proc = int(ipulse_global / npulses_local);
    if (proc != mpirank_optim) return;

    /* Open file  */
    FILE *file_c;
    file_c = fopen(filename, "w");

    /* Iterate over time points, compute expected energy and write to file. */
    int ntime = data[0][0].size(); // number of time points in the data
    for (int i=0; i<ntime; i++){
      double time = tstart + i*dt;

      Vec x = getData(time, ipulse_global, iinit_global);
      if (x != NULL) {
        double val = expectedEnergy(x, lindbladtype, nlevels, ioscillator);
        fprintf(file_c, "% 1.8f   % 1.14e   \n", time, val);
      }
    }
    fclose(file_c);
    // printf("%d: File written: %s\n", mpirank_optim, filename);
  }
}


void Data::writeFullstate(const char* filename_re, const char* filename_im, int ipulse_global, int iinit_global){
  // NOTE: only possible to save the full state if the data holds density matrix data
  if (densityData){
    // Only the processor who owns this pulse trajectory should be writing the file, other procs exit here.
    int proc = int(ipulse_global/ npulses_local);
    if (proc != mpirank_optim) return;

    /* Open files  */
    FILE *file_re, *file_im;
    file_re = fopen(filename_re, "w");
    file_im = fopen(filename_im, "w");

    /* Iterate over time points ane write vectorized state to file. */
    int ntime = data[0][0].size(); // number of time points in the data
    for (int i=0; i<ntime; i++){
      double time = tstart + i*dt;

      Vec x = getData(time, ipulse_global, iinit_global);
      if (x != NULL) {

        // write time in first column
        fprintf(file_re, "%1.8f  ", time);
        fprintf(file_im, "%1.8f  ", time);
        // write vectorized state in the following columns
        int size = 0;
        VecGetSize(x, &size);
        const PetscScalar *xptr;
        VecGetArrayRead(x, &xptr);
        for (int i=0; i<int(size/2); i++) {
          fprintf(file_re, "%1.10e  ", xptr[getIndexReal(i)]);  
          fprintf(file_im, "%1.10e  ", xptr[getIndexImag(i)]);  
        }
        fprintf(file_re, "\n");
        fprintf(file_im, "\n");
        VecRestoreArrayRead(x, &xptr);
      }
    }

    fclose(file_re);
    fclose(file_im);
    // printf("%d: File written: %s\n", mpirank_optim, filename_re);
    // printf("%d: File written: %s\n", mpirank_optim, filename_im);
  }
}

void Data::loadData_SyntheticQuandaryRho(std::vector<std::vector<std::string>> &data_names){
  // NOTE: This function is specialized to reading Re/Im density matrix data generate with Quandary
  densityData = true;

  // Iterate over local pulses
  for (int ipulse_local = 0; ipulse_local < npulses_local; ipulse_local++){
    int ipulse_global = mpirank_optim* npulses_local + ipulse_local;

    // Grab the data name vector for this pulse
    std::vector<std::string> data_name = data_names[ipulse_global];

    /* Extract control amplitudes from file name (searching for "p" and "q")*/
    std::size_t found_p = data_name[0].find("ctrlP");
    std::size_t found_q = data_name[0].find("ctrlQ");

    int strlength_p = 5;
    int strlength_q = 5;
    // If controls are given, load them, otherwise leave controlparams[ipulse] empty
    controlparams[ipulse_local].clear();
    if (found_p != std::string::npos && found_q != std::string::npos ){
      if (data_name[0][found_p+5] == '-') strlength_p=6;
      if (data_name[0][found_q+5] == '-') strlength_q=6;
      double p_Volt = std::stod(data_name[0].substr(found_p+5, strlength_p));
      double q_Volt = std::stod(data_name[0].substr(found_q+5, strlength_q));
      double conversion_factor = 1.0;  // conversion factor: Volt to GHz
      double p_GHz = p_Volt * conversion_factor;
      double q_GHz = q_Volt * conversion_factor;
      // printf("Got the control amplitudes %1.8f,%1.8f GHz\n", p_GHz, q_GHz);
      controlparams[ipulse_local].push_back(p_GHz);
      controlparams[ipulse_local].push_back(q_GHz);
    }

    // Iterate over local initial conditions 
    for (int iinit_local=0; iinit_local< ninit_local; iinit_local++){
      int iinit_global = mpirank_init * ninit_local + iinit_local;

      // Open data file for this initial condition and this pulse number
      std::ifstream infile_re;
      std::ifstream infile_im;
      infile_re.open(data_name[2*iinit_global + 0], std::ifstream::in);
      infile_im.open(data_name[2*iinit_global + 1], std::ifstream::in);
      if(infile_re.fail() ) {
          std::cout << "\n " << mpirank_world<< ": ERROR loading learning data file " << data_name[2*iinit_global + 0] << std::endl;
          exit(1);
      } else if (infile_im.fail() ) {// checks to see if file opended 
          std::cout << "\n "<< mpirank_world<< ": ERROR loading learning data file " << data_name[2*iinit_global + 1] << std::endl;
          exit(1);
      } else {
        std::cout<< "rank = " << mpirank_world<< ": Loading synthetic density data from " << data_name[2*iinit_global + 0] << ", " << data_name[2*iinit_global + 1] << std::endl;
      }

      // Iterate over each line in the files
      int count = 0;
      double time_re, time_im, time_prev;
      while (infile_re >> time_re) 
      {
        // Set the time and dt
        if (count == 0) tstart = time_re;
        if (count == 1) dt = time_re - time_im; // Note: since 'time_re' is read in the 'while' statement, it will have value from the 2nd row here, whereas time_im still has the value from the first row, hence dt = re - im
        infile_im >> time_im; // Read in time for the im file (it's already done for re in the while statement!)
        // printf("time_re = %1.8f, time_im = %1.8f\n", time_re, time_im);
        assert(fabs(time_re - time_im) < 1e-12);

        // printf("Loading data at Time %1.8f\n", time_re);
        // Break if exceeding the requested time domain length
        if (time_re > tstop)  {
          // printf("Stopping data at %1.8f > %1.8f \n", time_re, tstop);
          break;
        }

        // Now iterate over the remaining columns and store values.
        Vec state;
        VecCreate(PETSC_COMM_WORLD, &state);
        VecSetSizes(state, PETSC_DECIDE, 2*dim);
        VecSetFromOptions(state);
        double val_re, val_im;
        for (int i=0; i<dim; i++) { // Other elements are the state (re and im) at this time
          infile_re >> val_re;
          infile_im >> val_im;
          VecSetValue(state, getIndexReal(i), val_re, INSERT_VALUES);
          VecSetValue(state, getIndexImag(i), val_im, INSERT_VALUES);
        }
        VecAssemblyBegin(state);
        VecAssemblyEnd(state);

        // Store the state
        data[ipulse_local][iinit_local].push_back(state);  
        count+=1;
        time_prev = time_re;
      }

      /* Update the final time stamp */
      tstop = std::min(time_prev, tstop);

      // Close files
      infile_re.close();
      infile_im.close();
            
    }
  }

  // // TEST what was loaded
  // printf("\nDATA POINTS:\n");
  // for (int ipulse=0; ipulse<data.size(); ipulse++){
  //   printf("Control amplutidue: %f %f\n", controlparams[ipulse][0], controlparams[ipulse][1]);
  //     for (int j=0; j<data[ipulse][i].size(); i++){
  //       VecView(data[ipulse][i][j], NULL);
  //      }
  //   printf("\n");
  // }
  // printf("END DATA POINTS.\n\n");
  // printf("%d controlparams.size()=%d\n", mpirank_world, controlparams.size());
  // for (size_t i=0; i<controlparams.size(); i++){
  //   for (size_t j=0; j<controlparams[i].size(); j++){
  //     printf("%d controlparams[%d][%d] = %f\n", mpirank_world, i,j, controlparams[i][j]);
  //   }
  // }
  // exit(1);

  if (mpirank_world == 0) printf("-> Data loaded sucessfully. Data dt = %f, tstop = %f\n", dt, tstop);
}

void Data::loadData_SyntheticQuandaryPop(std::vector<std::vector<std::string>> &data_names){
  // NOTE: This function is specialized to reading synthetic population data generated with Quandary.
  densityData = false;


  // Iterate over local pulses
  for (int ipulse_local = 0; ipulse_local < npulses_local; ipulse_local++){
    int ipulse_global = mpirank_optim* npulses_local + ipulse_local;

    // Grab the data name vector for this pulse
    std::vector<std::string> data_name = data_names[ipulse_global];
    
    /* Extract control amplitudes from file name (searching for "p" and "q") */
    std::size_t found_p = data_name[0].find("ctrlP");
    std::size_t found_q = data_name[0].find("ctrlQ");

    int strlength_p = 5;
    int strlength_q = 5;
    // If controls are given, load them, otherwise leave controlparams[ipulse] empty
    controlparams[ipulse_local].clear();
    if (found_p != std::string::npos && found_q != std::string::npos ){
      if (data_name[0][found_p+5] == '-') strlength_p=6;
      if (data_name[0][found_q+5] == '-') strlength_q=6;
      double p_Volt = std::stod(data_name[0].substr(found_p+5, strlength_p));
      double q_Volt = std::stod(data_name[0].substr(found_q+5, strlength_q));
      double conversion_factor = 1.0;  // conversion factor: Volt to GHz
      double p_GHz = p_Volt * conversion_factor;
      double q_GHz = q_Volt * conversion_factor;
      // printf("Got the control amplitudes %1.8f,%1.8f GHz\n", p_GHz, q_GHz);
      controlparams[ipulse_local].push_back(p_GHz);
      controlparams[ipulse_local].push_back(q_GHz);
    }

    // Iterate over local initial conditions
    for (int iinit_local =0; iinit_local < ninit_local; iinit_local++){
      int iinit_global = mpirank_init * ninit_local + iinit_local;

      // Open the data file for this initial condition and this pulse number
      std::ifstream infile_pop;
      infile_pop.open(data_name[iinit_global], std::ifstream::in);
      if(infile_pop.fail() ) {
          std::cout << "\n " << mpirank_world << ": ERROR loading learning data file " << data_name[iinit_global] << std::endl;
          exit(1);
      } else {
        std::cout<< "rank = " << mpirank_world << ": Loading synthetic population data from " << data_name[iinit_global] << std::endl;
      }

      // Iterate over each line in the files
      int count = 0;
      double time_pop, time_prev;

      // read the header line
      std::string headerLine;
      std::getline(infile_pop, headerLine);
      // std::cout << "Header line: " << headerLine << std::endl;
      while (infile_pop >> time_pop) 
      {
        // Figure out time and dt
        if (count == 0){
          tstart = time_pop;
          // std::cout << "tstart = " << time_pop << std::endl;
        }
        if (count == 1){
          dt = time_pop - time_prev; // Assume a constant time step
          // std::cout << "time_prev = " << time_prev << std::endl;
          // std::cout << "dt = " << time_pop - time_prev << std::endl;
        } 

        // printf("Loading data at Time %1.8f\n", time_pop);
        // Break if exceeding the requested time domain length
        if (time_pop > tstop)  {
          // printf("Stopping data at %1.8f > %1.8f \n", time_pop, *tstop);
          break;
        }

        // Now iterate over the remaining columns and store values.
        Vec state;
        VecCreate(PETSC_COMM_WORLD, &state);
        VecSetSizes(state, PETSC_DECIDE, dim_hs); // Number of populations per time is dim_hs
        VecSetFromOptions(state);
        double val_pop;
        for (int i=0; i<dim_hs; i++) { // Remaining elements are the populations at this time level
          infile_pop >> val_pop;
          VecSetValue(state, i, val_pop, INSERT_VALUES);
        }
        VecAssemblyBegin(state);
        VecAssemblyEnd(state);

        // Store the state
        data[ipulse_local][iinit_local].push_back(state);
        count+=1;
        time_prev = time_pop;
      }

      /* Update the final time stamp */
      tstop = std::min(time_prev, tstop);

      // Close files
      infile_pop.close();            
    }
  }

  // // TEST what was loaded
  // printf("\nDATA POINTS:\n");
  // for (int ipulse=0; ipulse<data.size(); ipulse++){
  //   printf("Control amplutidue: %f %f\n", controlparams[ipulse][0], controlparams[ipulse][1]);
  //   for (int i=0; i<data[ipulse].size(); i++){
  //     for (int j=0; j<data[ipulse][i].size(); i++){
  //       VecView(data[ipulse][i][j], NULL);
  //      }
  //   }
  //   printf("\n");
  // }
  // printf("END DATA POINTS.\n\n");
  // exit(1);
  if (mpirank_world == 0) printf("-> Data loaded sucessfully. Data dt = %f, tstop = %f\n", dt, tstop);
}

// Tant2levelData::Tant2levelData(Config config, MPI_Comm comm_optim_, MPI_Comm comm_init_, int ninit_, std::vector<std::string>& data_name, std::vector<int> nlevels_, LindbladType lindbladtype_, bool densityData) : Data(config, comm_optim_, comm_init_, ninit_, data_name, nlevels_, lindbladtype_, densityData){

//   // Only for 2level data. 
//   assert(dim == 4);

//   /* Check whether data should be corrected or raw */
//   corrected = false;
//   if (data_name[0].compare("corrected") == 0) {
//     corrected = true;
//     data_name.erase(data_name.begin());
//   }

//   /* Load training data, this also sets first and last time stamp as well as data sampling step size */
//   loadData(&tstart, &tstop, &dt);

//   /* Set pulse amplitude. Hardcoded here. TODO. */
//   // Those are taken from 231110_SG_Tant_2level_constAndRandompulse_raw_and_corrected_2000shots/const_pulse/*_const_ampfac*_popt_rs*.dat
//   std::vector<std::vector<double>> datacontrols;
//   datacontrols.resize(5);
//   std::vector<double> amp{0.00177715, 0.00355431, 0.00533146, 0.00710861, 0.00888577};
//   for (int i=0; i<5; i++) {
//     datacontrols[i].push_back( amp[i]*1e3/(2.0*M_PI));
//     datacontrols[i].push_back(-amp[i]*1e3/(2.0*M_PI)); // NOT SURE WHY -q here, but that seems to match
//   }
//   // Now distribute them over optim processors
//   for (int ipulse_local=0; ipulse_local < npulses_local; ipulse_local++){
//     int ipulse_global = mpirank_optim*npulses_local + ipulse_local;
//     controlparams[ipulse_local].clear();
//     for (int ip=0; ip<datacontrols[ipulse_global].size(); ip++) {
//       controlparams[ipulse_local].push_back(datacontrols[ipulse_global][ip]);
//     }
//   }

//   if (mpirank_world == 0) {
//     printf("Training data in [%1.4f, %1.4f] us, sampling rate dt=%1.4f us\n", tstart, tstop, dt);
//     printf("Training data with constant controls\n");
//   }
// }

// Tant2levelData::~Tant2levelData(){}

// void Tant2levelData::loadData(double* tstart, double* tstop, double* dt){

//   /* Data format: First row is header following rows are formated as follows: 
//   *   <nshots> <time [us]> <pulse_num> <rho_lie_ij> for ij=1,2 <rho_lie_phys_ij> for ij=12 
//   *    int     double   int       (val_re+val_imj)              (val_re+val_imj)
//   */

//   // Iterate over local pulses
//   for (int ipulse_local = 0; ipulse_local < npulses_local; ipulse_local++){
//     int ipulse_global = mpirank_optim* npulses_local + ipulse_local;

//     // Open the respective file 
//     std::ifstream infile;
//     infile.open(data_name[ipulse_global], std::ifstream::in);
//     if(infile.fail() ) {// checks to see if file opended 
//         std::cout << "\n ERROR loading learning data file " << data_name[ipulse_global] << std::endl;
//         exit(1);
//     } else {
//       std::cout<< mpirank_world << ": Loading Tant Device data from " << data_name[ipulse_global] << std::endl;
//     }

//     // Iterate over lines
//     int count = 0;
//     double time, time_prev;
//     int pulse_num;
//     std::string strval;
//     std::string tmp; 
//     while (infile >> tmp) 
//     {
//       // Skip header lines
//       if (tmp.compare("Nshots") == 0){
//         for (int i=0; i< 300; i++){
//           infile >> tmp;
//           if (tmp.compare("rho_lie_phys_11") == 0) { // this is the last word in the first row
//             infile >> nshots; // nshots from the next line.
//             break;
//           }
//         }
//       }
//       nshots = std::atoi(tmp.c_str());

//       infile >> time;      // 2nd column;
//       infile >> pulse_num; // 3th column;
//       assert(pulse_num == ipulse_global);

//       // Figure out first time point and sampling time-step
//       if (count == 0) *tstart = time;
//       if (count == 1) *dt = time - time_prev; 
//       // printf("tstart = %1.8f\n", *tstart);
//       // printf("Loading data at Time %1.8f\n", time);

//       // Break if exceeding the requested time domain length 
//       if (time > *tstop)  {
//         if (npulses == 1) {
//           break; 
//         } else { // Skip to next pulse number.
//           while (infile >> tmp) {
//             if (tmp.compare("rho_lie_phys_11") == 0) {
//               break;
//             }
//           }
//           continue;
//         }
//       }
    
//       // Skip to the corrected physical LIE data (skip next <dim> columns)
//       if (corrected) {
//         for (int i=0; i < dim; i++){
//           infile >> strval;
//         }
//       }
    
//       // Allocate the state 
//       Vec state;
//       VecCreate(PETSC_COMM_WORLD, &state);
//       VecSetSizes(state, PETSC_DECIDE, 2*dim);
//       VecSetFromOptions(state);
 
//       // Iterate over the remaining columns and store values.
//       // Note, the vectorization of the density matrix in Yujin's Tant 2level data is row-wise whereas in quandary is columnwise, so here is the mapping of idices
//       std::vector<int> ids{0,2,1,3};
//       for (int i=0; i<dim; i++) {
//         // Read string and remove the brackets around (val_re+val_imj) as well as the trailing "j" character
//         infile >> strval;
//         std::string stripped = strval.substr(1, strval.size()-3);  
//         // Extract real and imaginary parts, with correct sign.
//         std::stringstream stream(stripped); 
//         double val_re, val_im;
//         char sign;
//         while (stream >> val_re >> sign >> val_im) {
//           if (sign == '-') val_im = -val_im;
//           // std::cout<< " -> Got val = " << val_re << " " << val_im << " j" << std::endl;
//         }
//         VecSetValue(state, getIndexReal(ids[i]), val_re, INSERT_VALUES);
//         VecSetValue(state, getIndexImag(ids[i]), val_im, INSERT_VALUES);
//       }
//       VecAssemblyBegin(state);
//       VecAssemblyEnd(state);

//       // Store the state and update counters
//       // Note: only one initial condition per pulse. 
//       data[ipulse_local][0].push_back(state);
//       count+=1;
//       time_prev = time;

//       // Skip to the end of the line, if we used non-corrected data
//       if (!corrected) {
//         for (int i=0; i < dim; i++){
//           infile >> strval;
//         }
//       }
//     }

//     /* Update the final time stamp */
//     *tstop = std::min(time_prev, *tstop);

//     // Close files
//     infile.close();
//   }
// }

// Tant3levelData::Tant3levelData(Config config, MPI_Comm comm_optim_, MPI_Comm comm_init_, int ninit_, std::vector<std::string>& data_name, std::vector<int> nlevels_, LindbladType lindbladtype_, bool densityData) : Data(config, comm_optim_, comm_init_, ninit_, data_name, nlevels_, lindbladtype_, densityData) {
//   // Only for 3level data. 
//   assert(dim == 9);

//   /* Check whether data should be corrected or raw */
//   corrected = false;
//   if (data_name[0].compare("corrected") == 0) {
//     corrected = true;
//     data_name.erase(data_name.begin());
//   }

//   /* Load training data */
//   loadData(&tstart, &tstop, &dt);
// }

// Tant3levelData::~Tant3levelData(){}

// void Tant3levelData::loadData(double* tstart, double* tstop, double* dt){

//   /* Data format: First row is header following rows are probabilities of the identity and 8 rotation operators
//   *   <line> | <time [ns]> | <P(R_i=j)> for i=0,...8, j=0,1,2, | mitigated P(R_i=j)
//   *    int     double        double [...]                      | double [...]
//   */

//   // Dimension of the Hilbert space (N);
//   // int dim_rho = int(sqrt(dim));   // dim_rho = 3, dim = 9 // Moved to Data constructor, now dim_hs

//   // Number of pulses
//   assert(npulses = data_name.size());

//   // Iterate over local pulses
//   for (int ipulse_local = 0; ipulse_local < npulses_local; ipulse_local++){
//     int ipulse = mpirank_optim* npulses_local + ipulse_local;
  
//     /* Extract control amplitudes from file name */
//     double conversion_factor = 47.90850565409482;  // conversion factor: Volt to MHz
//     std::size_t found_p = data_name[ipulse].find_last_of("p");
//     std::size_t found_q = data_name[ipulse].find_last_of("q");
//     int strlength_p = 5;
//     int strlength_q = 5;
//     if (data_name[ipulse][found_p+1] == '-') strlength_p=6;
//     if (data_name[ipulse][found_q+1] == '-') strlength_q=6;
//     double p_Volt = std::stod(data_name[ipulse].substr(found_p+1, strlength_p));
//     double q_Volt = std::stod(data_name[ipulse].substr(found_q+1, strlength_q));

//     if      (p_Volt == 0.015) p_Volt = 0.014797339327183157;
//     else if (p_Volt ==-0.015) p_Volt =-0.014797339327183157;
//     else if (p_Volt == 0.018) p_Volt = 0.018122965451208638;
//     else if (p_Volt ==-0.018) p_Volt =-0.018122965451208638;
//     else if (p_Volt == 0.026) p_Volt = 0.02562974353151829;
//     else if (p_Volt ==-0.026) p_Volt =-0.02562974353151829;
//     if      (q_Volt == 0.015) q_Volt = 0.014797339327183157;
//     else if (q_Volt ==-0.015) q_Volt =-0.014797339327183157;
//     else if (q_Volt == 0.018) q_Volt = 0.018122965451208638;
//     else if (q_Volt ==-0.018) q_Volt =-0.018122965451208638;
//     else if (q_Volt == 0.026) q_Volt = 0.02562974353151829;
//     else if (q_Volt ==-0.026) q_Volt =-0.02562974353151829;

//     double p_MHz = p_Volt * conversion_factor;
//     double q_MHz = q_Volt * conversion_factor;
//     // printf("Got the control amplitudes %1.8f,%1.8f MHz\n", p_MHz, q_MHz);
//     controlparams[ipulse_local].push_back(p_MHz);
//     controlparams[ipulse_local].push_back(q_MHz);
  
//     /* Open the data file */
//     std::ifstream infile;
//     infile.open(data_name[ipulse], std::ifstream::in);
//     if(infile.fail() ) {// checks to see if file opended 
//         std::cout << "\n " << mpirank_optim << ": ERROR loading learning data file " << data_name[ipulse] << std::endl;
//         exit(1);
//     } else {
//       std::cout<< mpirank_optim << ": Loading Tant3Level data from " << data_name[ipulse] << std::endl;
//     }

//     /* Skip first line, it's just the header. */
//     std::string tmp; 
//     for (int i=0; i< 300; i++){
//       infile >> tmp;
//       if (tmp.compare("Proj_op8_state2_mitigated") == 0) { // this is the last word in the first row
//         break;
//       }
//     }

//     /* Now read each column */
//     double time, time_prev;
//     std::string strval;
//     int count = 0;
//     int linenumber = 1;
//     while (infile >> linenumber || count < 1) {
//       infile >> time ;      // 2nd column

//       // Time in file is ns, scale to us here:
//       time = time * 1e-3; // us

//       // Figure out first time point and sampling time-step
//       if (count == 0) *tstart = time;
//       if (count == 1) *dt = time - time_prev; 
//       // printf("tstart = %1.8f, dt=%1.8f\n", tstart, data_dt);
//       // printf("Loading data at Time %1.8f tstop = %1.8f\n", time, *tstop);

//       // Break if exceeding the requested time domain length 
//       if (time > *tstop) break; 

//       // Skip to the corrected (mitigated) data (skip next 9*3=27 columns)
//       if (corrected) {
//         for (int i=0; i < dim*dim_hs; i++){
//           infile >> strval;
//         }
//       }

//       // Allocate the state 
//       Vec state;
//       VecCreate(PETSC_COMM_WORLD, &state);
//       VecSetSizes(state, PETSC_DECIDE, 2*dim);
//       VecSetFromOptions(state);

//       // Iterate over the remaining columns to read the probabilities of the rotation operators R_0 to R_8
//       double val;
//       std::vector<std::vector<double>> prob;  // outer dimension for i, inner for j
//       prob.resize(dim_hs*dim_hs);
//       for (int i=0; i<dim_hs*dim_hs; i++) { // operators R_i
//         prob[i].resize(3);
//         for (int j=0; j<dim_hs; j++) { // outcomes R_i = j, j=0,1,2
//           infile >> prob[i][j];  // probability P(R_i = j)
//           // printf("Read P(R %d = %d) = %1.4e\n", i, j, prob[i][j]);

//         }
//       }

//       // Correct the probabilities: Clip to [0,1] and sum = 1.0
//       if (corrected) {
//         for (int i=0; i<prob.size(); i++){
//           double sum = 0.0;
//           for (int j=0; j<prob[i].size(); j++){ 
//             // Clip to [0,1]
//             prob[i][j] = std::max(0.0, prob[i][j]);  
//             prob[i][j] = std::min(1.0, prob[i][j]);
//             sum += prob[i][j];
//           }
//           // Make sure they sum to 1.0 by reducing 2nd state probability
//           prob[i][2] -= (sum - 1.0);
//           sum = 0.0;
//           for (int j=0; j<dim_hs; j++){
//             sum += prob[i][j];
//           }
//         }
//       }

//       // Now assemble the coefficients r_k of the Gellmann basis expansion \rho = 1/N Id + \sum_{k=1,8}}r_k sigma_k
//       double r1 = -prob[2][0] + prob[2][1]; 
//       double r2 = -prob[1][1] + prob[1][0]; 
//       double r3 = -prob[0][1] + prob[0][0]; 
//       double r4 = -prob[5][0] + prob[5][2]; 
//       double r5 = -prob[4][2] + prob[4][0]; 
//       double r6 = -prob[7][1] + prob[7][2]; 
//       double r7 = -prob[6][2] + prob[6][1]; 
//       double r8 = prob[0][0]/sqrt(3) + prob[0][1]/sqrt(3) - 2.0/sqrt(3)*prob[0][2];

//       // Now assemble the NxN density matrix
//       Mat rho_re, rho_im; 
//       MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_hs, dim_hs, NULL, &rho_re);
//       MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim_hs, dim_hs, NULL, &rho_im);
//       MatSetUp(rho_re);
//       MatSetUp(rho_im);
//       MatAssemblyBegin(rho_re, MAT_FINAL_ASSEMBLY);
//       MatAssemblyBegin(rho_im, MAT_FINAL_ASSEMBLY);
//       MatAssemblyEnd(rho_re, MAT_FINAL_ASSEMBLY);
//       MatAssemblyEnd(rho_im, MAT_FINAL_ASSEMBLY);
//       std::vector<Mat> BasisMat_Re;
//       std::vector<Mat> BasisMat_Im;
//       createGellmannMats(dim_hs, false, false, false, true, BasisMat_Re, BasisMat_Im);
//       // Note: The Gellmann matrices are ordered in a different way than the above coefficients. Too bad... Here is the mapping. 
//       MatAXPY(rho_re, 1.0/dim_hs, BasisMat_Re[0], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_re, r1/2.0, BasisMat_Re[1], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_re, r4/2.0, BasisMat_Re[2], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_re, r6/2.0, BasisMat_Re[3], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_re, r3/2.0, BasisMat_Re[4], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_re, r8/2.0, BasisMat_Re[5], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_im, r2/2.0, BasisMat_Im[0], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_im, r5/2.0, BasisMat_Im[1], DIFFERENT_NONZERO_PATTERN);
//       MatAXPY(rho_im, r7/2.0, BasisMat_Im[2], DIFFERENT_NONZERO_PATTERN);

//       // Now vectorize the density matrix and store into the state 
//       for (int col = 0; col < dim_hs; col++){
//         PetscScalar *vals_re, *vals_im; 
//         MatDenseGetColumn(rho_re, col, &vals_re);
//         MatDenseGetColumn(rho_im, col, &vals_im);
//         for (int i=0; i<dim_hs; i++){
//           int row = col*dim_hs +i;
//           VecSetValue(state, getIndexReal(row), vals_re[i], INSERT_VALUES);
//           VecSetValue(state, getIndexImag(row), vals_im[i], INSERT_VALUES);
//         }
//         MatDenseRestoreColumn(rho_re, &vals_re);
//         MatDenseRestoreColumn(rho_im, &vals_im);
//       }
//       VecAssemblyBegin(state);
//       VecAssemblyEnd(state);
 
//       // Cleanup
//       MatDestroy(&rho_re);
//       MatDestroy(&rho_im);

//       // Store the state
//       // Note: only one initial condition per pulse.
//       data[ipulse_local][0].push_back(state);  // Here, only one pulse
//       count+=1;
//       time_prev = time;

//       // Skip to the end of file, if we used non-corrected data
//       if (!corrected) {
//         for (int i=0; i < dim*dim_hs; i++){
//           infile >> strval;
//         }
//       }
//     }

//     /* Update the final time stamp */
//     *tstop = std::min(time_prev, *tstop);

//     // Close files
//     infile.close();
//   }

//   // // TEST what was loaded
//   // printf("\nDATA POINTS:");
//   // for (int ipulse=0; ipulse<data.size(); ipulse++){
//   //   printf("PULSE NUMBER %d\n", ipulse);
//   //   for (int j=0; j<data[ipulse].size(); j++){
//   //     VecView(data[ipulse][j], NULL);
//   //   }
//   // }
//   // printf("END DATA POINTS. tstart = %1.8f, tstop=%1.8f, dt=%1.8f\n", *tstart, *tstop, *dt);
//   // exit(1);

//   if (mpirank_world == 0) printf("Training data in [%1.4f, %1.4f] us, sampling rate dt=%1.4f us\n", *tstart, *tstop, *dt);
//   if (mpirank_world == 0) printf("Training data with constant controls\n");
// }