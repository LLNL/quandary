#include <data.hpp>

Data::Data() {
	dt = 0.0;
  dim = -1;
}

Data::Data(std::vector<std::string> data_name_, int dim_) {
  data_name = data_name_;
  dim = dim_;

}

Data::~Data() {
  for (int i=0; i<data.size(); i++){
    VecDestroy(&data[i]);
  }
  data.clear();
}

SyntheticQuandaryData::SyntheticQuandaryData(std::vector<std::string> data_name, double data_tstop, int dim) : Data(data_name, dim) {

  /* Load training data */
  dt = loadData(data_tstop);
}

SyntheticQuandaryData::~SyntheticQuandaryData() {

}

double SyntheticQuandaryData::loadData(double tstop) {

  // Open files 
  std::ifstream infile_re;
  std::ifstream infile_im;
  infile_re.open(data_name[0], std::ifstream::in);
  infile_im.open(data_name[1], std::ifstream::in);
  if(infile_re.fail() || infile_im.fail() ) {// checks to see if file opended 
      std::cout << "\n ERROR loading learning data file\n" << std::endl; 
      std::cout << data_name[0] + "_re.dat" << std::endl;
      exit(1);
  } else {
    std::cout<< "Loading synthetic data from " << data_name[0] << ", " << data_name[1] << std::endl;
  }

  // Iterate over each line in the files
  int count = 0;
  double time_re, time_im;
  double data_dt;
  while (infile_re >> time_re) 
  {
    // Figure out time and dt
    if (count == 1) {
      data_dt = time_re - time_im; // Note: since 'time_re' is read in the 'while' statement, it will have value from the 2nd row here, whereas time_im still has the value from the first row, hence dt = re - im
    } 
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
    data.push_back(state);
    count+=1;
  }

  // Close files
	infile_re.close();
	infile_im.close();

  // // TEST what was loaded
  // printf("\nDATA POINTS:\n");
  // for (int i=0; i<data.size(); i++){
  //   VecView(data[i], NULL);
  // }
  // printf("END DATA POINTS.\n\n");
  return data_dt;
}
