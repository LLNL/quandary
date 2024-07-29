#include <data.hpp>

Data::Data() {
	ntime = 0;
	dt = 0.0;
  dim = -1;
}

Data::Data(std::vector<std::string> data_name_, double data_dt_, int data_ntime_, int dim_) {
	ntime = data_ntime_;
	dt = data_dt_;
  data_name = data_name_;
  dim = dim_;

}

Data::~Data() {
  for (int i=0; i<data.size(); i++){
    VecDestroy(&data[i]);
  }
  data.clear();
}

SyntheticQuandaryData::SyntheticQuandaryData(std::vector<std::string> data_name, double data_dt, int data_ntime, int dim) : Data(data_name, data_dt, data_ntime, dim) {

  loadData();
}

SyntheticQuandaryData::~SyntheticQuandaryData() {

}

void SyntheticQuandaryData::loadData() {

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
    std::cout<< "Loading trajectory data from " << data_name[0] << ", " << data_name[1] << std::endl;
  }

  // Iterate over each line in the file
  int count = 0;
  double val_re = -1.0, val_im=-1.0;
  while (infile_re >> val_re) 
{
  for (int n = 0; n <ntime; n++) {
    Vec state;
    VecCreate(PETSC_COMM_WORLD, &state);
    VecSetSizes(state, PETSC_DECIDE, 2*dim);
    VecSetFromOptions(state);

    // Iterate over columns
    // double val_re, val_im;
    // infile_re >> val_re; // first element is time, but its already in read
    infile_im >> val_im; // first element is time
    // printf("time-step %1.4f == %1.4f ??\n", val_re, val_im);
    assert(fabs(val_re - val_im) < 1e-12);
    for (int i=0; i<dim; i++) { // Other elements are the state (re and im) at this time
      infile_re >> val_re;
      infile_im >> val_im;
      VecSetValue(state, getIndexReal(i), val_re, INSERT_VALUES);
      VecSetValue(state, getIndexImag(i), val_im, INSERT_VALUES);
    }
    VecAssemblyBegin(state);
    VecAssemblyEnd(state);
    data.push_back(state);
    count+=1;
  }
  ntime = count;

  // Close files
	infile_re.close();
	infile_im.close();

  // // TEST what was loaded
  // printf("\nDATA POINTS:\n");
  // for (int i=0; i<data.size(); i++){
  //   VecView(data[i], NULL);
  // }
  // printf("END DATA POINTS.\n\n");
}
