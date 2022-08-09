#include "pythoninterface.hpp"

PythonInterface::PythonInterface(){
}


PythonInterface::PythonInterface(std::string python_file, LindbladType lindbladtype_, int dim_rho_) {

  lindbladtype = lindbladtype_;
  dim_rho = dim_rho_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

#ifdef WITH_PYTHON

  // Set PYTHONPATH to working directory . TODO: CHECK if that's what you want!!
  setenv("PYTHONPATH", ".",1);

  // Initialize Python interpreter
  Py_Initialize();

  // Remove the ".py" from the file name to get the module name
  if (python_file.find(".py") != std::string::npos) python_file.erase(python_file.find(".py"), 3);

  /* Import and store the python script as a module */
  PyObject *pModuleName =  PyString_FromString(python_file.c_str());
  pModule = PyImport_Import(pModuleName);
  PyErr_Print();
  Py_DECREF(pModuleName); // TODO: Read about decrementing py_pointers (borrowed) and check if this is correct!

  if (pModule == NULL) {
    printf("ERROR importing python module %s.\n", python_file.c_str());
    exit(1);
  }
#endif
}

PythonInterface::~PythonInterface(){
#ifdef WITH_PYTHON
  Py_Finalize();
#endif
}

void PythonInterface::receiveHd(Mat& Ad, Mat& Bd){
#ifdef WITH_PYTHON

  if (mpirank_world == 0) printf("Receiving system Hamiltonian...\n");

  /* ----- Real system matrix Hd_real, stored in Bd ---- */

  // Get a pointer to the python function "getHd"
  PyObject* pFunc_getHd_real = PyObject_GetAttrString(pModule, (char*)"getHd"); 

  // Call the python function.
  PyObject *pHd;
  if (pFunc_getHd_real && PyCallable_Check(pFunc_getHd_real)) {
    pHd = PyObject_CallObject(pFunc_getHd_real, NULL); // NULL: no input to getHd().
    PyErr_Print();
  } else PyErr_Print();

  /* Convert the result from python object to double vector */
  int length = 0;
  if (PyList_Check(pHd)) {
    length = PyList_Size(pHd);
    PyErr_Print();
  }
  // printf("getHd(): Received a list of length = %d \n", length);
  std::vector<double> vals;
  std::vector<int> ids;
  for (Py_ssize_t i=0; i<length; i++){
    PyObject* value = PyList_GetItem(pHd,i);  // TODO: Does this need a Py_DECREF at some point?
    PyErr_Print();
    double val = PyFloat_AsDouble(value);
    PyErr_Print();
    if (fabs(val) > 1e-14) {  // store only nonzeros
      vals.push_back(val);
      ids.push_back(i);
    }
  }
  // print vals 
  // for (int i=0; i<vals.size(); i++) printf("%d %f, ", ids[i], vals[i]);
  // printf("\n");

  // Cleanup
  Py_XDECREF(pFunc_getHd_real); 

  /* Get matrix size */ 
  PetscInt dim = 0;
  MatGetSize(Bd, &dim, NULL); // could be N^2 or N

  /* Write values into sparse Petsc matrix Bd. */
  // Bd had been allocated before. Destroy and reallocate it here. 
  MatDestroy(&Bd);
  MatCreate(PETSC_COMM_WORLD, &Bd);
  MatSetType(Bd, MATMPIAIJ);
  MatSetSizes(Bd, PETSC_DECIDE, PETSC_DECIDE, dim, dim); // dim = N^2 for Lindblad, N for Schroedinger
  MatSetUp(Bd);
  MatSetFromOptions(Bd);
  PetscInt ilow, iupp;
  MatGetOwnershipRange(Bd, &ilow, &iupp);

  int sqdim = dim; // could be N^2 or N
  if (lindbladtype != LindbladType::NONE) sqdim = (int) sqrt(dim); // sqdim = N 

  // // TEST: Print out B_d without the I\kron Bd stuff
  // Mat Bd_test;
  // MatCreate(PETSC_COMM_WORLD, &Bd_test);
  // MatSetType(Bd_test, MATMPIAIJ);
  // MatSetSizes(Bd_test, PETSC_DECIDE, PETSC_DECIDE, sqdim, sqdim); 
  // MatSetUp(Bd_test);
  // MatSetFromOptions(Bd_test);

  // Iterate over nonzero elements
  for (int i = 0; i<ids.size(); i++) {
    // Get position in the Bd matrix
    int row = ids[i] % sqdim;
    int col = ids[i] / sqdim;
    // MatSetValue(Bd_test, row, col, vals[i], INSERT_VALUES);

    // If Schroedinger: Assemble -B_d 
    if (lindbladtype == LindbladType::NONE) {
      double val = -1.*vals[i];
      if (ilow <= row && row < iupp) MatSetValue(Bd, row, col, val, ADD_VALUES);
    } else {
    // If Lindblad: Assemble -I_N \kron B_d + B_d \kron I_N
      for (int k=0; k<sqdim; k++){
        // first place all -v_ij in the -I_N \kron B_d term:
        int rowk = row + sqdim * k;
        int colk = col + sqdim * k;
        double val = -1.*vals[i];
        if (ilow <= rowk && rowk < iupp) MatSetValue(Bd, rowk, colk, val, ADD_VALUES);
        // Then add v_ij in the B_d \kron I_N term:
        rowk = row*sqdim + k;
        colk = col*sqdim + k;
        val = vals[i];
        if (ilow <= rowk && rowk < iupp) MatSetValue(Bd, rowk, colk, val, ADD_VALUES);
      }
    } 
  }
  MatAssemblyBegin(Bd, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Bd, MAT_FINAL_ASSEMBLY);
  // MatAssemblyBegin(Bd_test, MAT_FINAL_ASSEMBLY);
  // MatAssemblyEnd(Bd_test, MAT_FINAL_ASSEMBLY);

  // // Write the test matrix to a file
  // PetscViewer viewer;
  // PetscViewerCreate(PETSC_COMM_SELF, &viewer); 
  // PetscViewerSetType(viewer, PETSCVIEWERASCII); 
  // PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); 
  // PetscViewerFileSetName(viewer, "Bd_test.txt");
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);
  // MatView(Bd_test, viewer);


  /* ----- Imaginary system matrix Hd_imag, stored in Ad.  ---- */
  // Note: Need to ADD to Ad rather than resetting it, since it might contain the Lindblad terms

  // Get a pointer to the python function "getHd_imag"
  PyObject* pFunc_getHd_imag = PyObject_GetAttrString(pModule, (char*)"getHd_imag"); 
  // Call the python function.
  PyObject *pHd_imag;
  bool called_imag = false;
  if (pFunc_getHd_imag && PyCallable_Check(pFunc_getHd_imag)) {
    pHd_imag = PyObject_CallObject(pFunc_getHd_imag, NULL); // NULL: no input to getHd().
    called_imag = true;
    PyErr_Print();
  } else PyErr_Print();

  if (!called_imag || !PyList_Check(pHd_imag)) {
    if (mpirank_world == 0) printf("# No imaginary system Hamiltonian received. \n");
  } else {

    /* Convert the result from python object to double vector */
    int length = 0;
    if (PyList_Check(pHd_imag)) {
      length = PyList_Size(pHd_imag);
      PyErr_Print();
    }
    // printf("getHd_imag(): Received a list of length = %d \n", length);
    std::vector<double> vals;
    std::vector<int> ids;
    for (Py_ssize_t i=0; i<length; i++){
      PyObject* value = PyList_GetItem(pHd_imag,i);  // TODO: Does this need a Py_DECREF at some point?
      PyErr_Print();
      double val = PyFloat_AsDouble(value);
      PyErr_Print();
      if (fabs(val) > 1e-14) {  // store only nonzeros
        vals.push_back(val);
        ids.push_back(i);
      }
    }
    // print vals 
    // for (int i=0; i<vals.size(); i++) printf("%d %f, ", ids[i], vals[i]);
    // printf("\n");

    // Cleanup
    Py_XDECREF(pFunc_getHd_imag); 

    /* Write values into sparse Petsc matrix Ad. */
    // Ad had been allocated before and potentially contains the Lindblad terms. Copy values into temporary matrix and add it later to what we read from the python file here. 
    Mat Ad_copy;
    MatDuplicate(Ad, MAT_COPY_VALUES, &Ad_copy);
    MatDestroy(&Ad);
    MatCreate(PETSC_COMM_WORLD, &Ad);
    MatSetType(Ad, MATMPIAIJ);
    MatSetSizes(Ad, PETSC_DECIDE, PETSC_DECIDE, dim, dim); // dim = N^2 for Lindblad, N for Schroedinger
    MatSetUp(Ad);
    MatSetFromOptions(Ad);
    MatGetOwnershipRange(Ad, &ilow, &iupp);

    sqdim = dim; // could be N^2 or N
    if (lindbladtype != LindbladType::NONE) sqdim = (int) sqrt(dim); // sqdim = N 

    // Iterate over nonzero elements
    for (int i = 0; i<ids.size(); i++) {
      // Get position in the Bd matrix
      int row = ids[i] % sqdim;
      int col = ids[i] / sqdim;
      // MatSetValue(Bd_test, row, col, vals[i], INSERT_VALUES);

      // If Schroedinger: Assemble -B_d 
      if (lindbladtype == LindbladType::NONE) {
        double val = -1.*vals[i];
        if (ilow <= row && row < iupp) MatSetValue(Ad, row, col, val, ADD_VALUES);
      } else {
      // If Lindblad: Assemble -I_N \kron B_d + B_d \kron I_N
        for (int k=0; k<sqdim; k++){
          // first place all -v_ij in the -I_N \kron B_d term:
          int rowk = row + sqdim * k;
          int colk = col + sqdim * k;
          double val = -1.*vals[i];
          if (ilow <= rowk && rowk < iupp) MatSetValue(Ad, rowk, colk, val, ADD_VALUES);
          // Then add v_ij in the B_d \kron I_N term:
          rowk = row*sqdim + k;
          colk = col*sqdim + k;
          val = vals[i];
          if (ilow <= rowk && rowk < iupp) MatSetValue(Ad, rowk, colk, val, ADD_VALUES);
        }
      } 
    }
    MatAssemblyBegin(Ad, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ad, MAT_FINAL_ASSEMBLY);

    /* Potentially add Lindblad terms */
    MatAXPY(Ad, 1.0, Ad_copy, DIFFERENT_NONZERO_PATTERN);
    MatDestroy(&Ad_copy);
  }



#endif
}



void PythonInterface::receiveHc(int noscillators, std::vector<std::vector<Mat>>& Ac_vec, std::vector<std::vector<Mat>>& Bc_vec){
#ifdef WITH_PYTHON

  if (mpirank_world == 0) printf("Receiving control Hamiltonian terms...\n");

  /* Reset Ac_vec and Bc_vec (Keeping the outer vector intact) */
  for (int k=0; k<Ac_vec.size(); k++){
    for (int l=0; l<Ac_vec[k].size(); l++){
      if (Ac_vec[k][l] != NULL) MatDestroy(&(Ac_vec[k][l]));
    }
    Ac_vec[k].clear();
  }
  for (int k=0; k<Bc_vec.size(); k++){
    for (int l=0; l<Bc_vec[k].size(); l++){
      if (Bc_vec[k][l] != NULL) MatDestroy(&(Bc_vec[k][l]));
    }
    Bc_vec[k].clear();
  }

  // Set the number of control terms for each oscillator to zero here. Overwrite later below, if we receive control terms. 
  ncontrol_real.clear();
  ncontrol_imag.clear();
  for (int k=0; k<noscillators;k++){
    ncontrol_real.push_back(0);
    ncontrol_imag.push_back(0);
  }

  /* Get the dimensions right */
  int sqdim = dim_rho; //  N!
  int dim = dim_rho;
  if (lindbladtype !=LindbladType::NONE) dim = dim_rho*dim_rho;

  /* REAL PART */

  // Get a reference to the python function "getHc_real"
  PyObject* pFunc_getHc_real = PyObject_GetAttrString(pModule, (char*)"getHc_real");
  // Call the python function
  PyObject *pHc_real;
  bool called_real = false;
  if (pFunc_getHc_real){
    if (PyCallable_Check(pFunc_getHc_real)) {
      pHc_real = PyObject_CallObject(pFunc_getHc_real, NULL); // NULL: no input to getHc().
      PyErr_Print();
      called_real = true;
    } else PyErr_Print();
  } else PyErr_Print();

  if (!called_real || !PyList_Check(pHc_real)) {
    if (mpirank_world == 0) printf("# No real control Hamiltonian received. \n");
  } else {

    // Parse the result 
    // getHc() MUST return a python list of lists of lists of float elements for this to work:
    // for each oscillator k=0...Q-1: for each control term i=0...C^k-1: a list containing the flattened Hamiltonian Hc^k_i

    // Sanity check: the outer list should have Q == noscillators elements (each one being a list of Hamiltonians)
    int Q = 0;
    if (PyList_Check(pHc_real)) {
      Q = PyList_Size(pHc_real); 
      PyErr_Print();
    }
    if (Q != noscillators) {
     printf("Error parsing python function getHc_real(): It should contain an (outer) list of length %d, but did return a list of length %d.\n", noscillators, Q);
     exit(1);
    }
    // printf("getHc(): Received an (outer) list of length %d\n", Q);

    // Receive and store Hamiltonian values
    std::vector<std::vector<double>> Hc_re_vals; // vector of vector of Hamiltonian values
    std::vector<std::vector<int>> Hc_re_ids;  // vector of vector of Hamiltonian id

    // Iterate over oscillators
    for (Py_ssize_t k=0; k<noscillators; k++){

      // Check number of control terms for this oscillator
      int ncontrol = 0;
      PyObject* pHck_real = PyList_GetItem(pHc_real,k); // Get list of Hamiltonians for oscillator k
      PyErr_Print();
      if (PyList_Check(pHck_real)) { 
        ncontrol = PyList_Size(pHck_real); 
        PyErr_Print();
      } else PyErr_Print();
      ncontrol_real[k] = ncontrol;
      // printf("getHc_real(): For oscillator %d, received %d control Hamiltonians Re(Hc^k).\n", (int)k, ncontrol_real[k]);

      // Iterate over control terms for this oscillator 
      for (Py_ssize_t i=0; i<ncontrol_real[k]; i++){
        PyObject* pHcki_real = PyList_GetItem(pHck_real,i); // Get the Hamiltonian Hcki
        PyErr_Print();

        int Hcki_size_re = 0;
        if (PyList_Check(pHcki_real)) {
          Hcki_size_re = PyList_Size(pHcki_real);
          PyErr_Print();
        }
        int Hcki_size = Hcki_size_re;
        // printf("getHc(): Oscillator %d, term %i: Received a list of length = %d: ", (int)k,(int)i,Hcki_size);
        std::vector<double> Hcki_re_vals;
        std::vector<int> Hcki_re_ids;
        for (Py_ssize_t l=0; l<Hcki_size; l++){ // Iterate over elements
          PyObject* pval_re = PyList_GetItem(pHcki_real,l);
          PyErr_Print();
          double Hcki_re_val = PyFloat_AsDouble(pval_re);
          PyErr_Print();
          if (fabs(Hcki_re_val) > 1e-14) {  // store only nonzeros
            Hcki_re_vals.push_back(Hcki_re_val);
            Hcki_re_ids.push_back(l);
          }
        }
        Hc_re_vals.push_back(Hcki_re_vals);
        Hc_re_ids.push_back(Hcki_re_ids);
      } // end of control term i for this oscillator k
    } // end of oscillator k

    /* Now place values into Bc_vec[k][l] for all oscillators k and all control terms l */

    int ioscil = 0;  // index for accessing Hamiltonian values inside Hc_re_vals/ids
    for (int k=0; k<noscillators; k++){

      if (mpirank_world == 0) printf("Creating %d control Mats for oscillator %d\n", ncontrol_real[k], k);

      // Iterate over control terms for this oscillator
      for (int i=0; i<ncontrol_real[k]; i++){

        // Create a new control matrix
        Mat myBcMat_kl;
        Bc_vec[k].push_back(myBcMat_kl);

        MatCreate(PETSC_COMM_WORLD, &(Bc_vec[k][i]));
        MatSetType(Bc_vec[k][i], MATMPIAIJ);
        MatSetSizes(Bc_vec[k][i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
        MatSetUp(Bc_vec[k][i]);
        MatSetFromOptions(Bc_vec[k][i]);

        PetscInt ilow, iupp;
        MatGetOwnershipRange(Bc_vec[k][i], &ilow, &iupp);

        // Assemble -I_N \kron Hc^k_i + Hc^k_i \kron I_N (Lindblad) or -Hc^k_i (Schroedinger)
        // vals are in Hc_vals[ioscil][:]
        // Iterate over nonzero elements in Hc^k_i
        for (int l = 0; l<Hc_re_ids[ioscil].size(); l++) {
          // Get position in the Bc matrix
          int row = Hc_re_ids[ioscil][l] % sqdim;
          int col = Hc_re_ids[ioscil][l] / sqdim;

          if (lindbladtype == LindbladType::NONE){
            // Schroedinger
            // Assemble - B_c  
            double val = -1.*Hc_re_vals[ioscil][l];
            if (ilow <= row && row < iupp) MatSetValue(Bc_vec[k][i], row, col, val, ADD_VALUES);
          } else {
            // Lindblad
            // Assemble -I_N \kron B_c + B_c \kron I_N 
            for (int m=0; m<sqdim; m++){
              // first place all -v_ij in the -I_N\kron B_c term:
              int rowm = row + sqdim * m;
              int colm = col + sqdim * m;
              double val = -1.*Hc_re_vals[ioscil][l];
              if (ilow <= rowm && rowm < iupp) MatSetValue(Bc_vec[k][i], rowm, colm, val, ADD_VALUES);
              // Then add v_ij in the B_d^T \kron I_N term:
              rowm = col*sqdim + m;   // transpose!
              colm = row*sqdim + m;
              val = Hc_re_vals[ioscil][l];
              if (ilow <= rowm && rowm < iupp) MatSetValue(Bc_vec[k][i], rowm, colm, val, ADD_VALUES);
            }
          }
        } // end of elements in Hc^k_i
        ioscil++;
      } // end of i loop for control terms

      // Assemble the matrices for this oscillator
      for (int i=0; i<ncontrol_real[k]; i++){
        MatAssemblyBegin(Bc_vec[k][i], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Bc_vec[k][i], MAT_FINAL_ASSEMBLY);
      }
    } // end of k loop for oscillators for REAL Hamiltonian
  } // end of reading real part from python

  /* IMAGINARY PART */

  // Get a reference to the python function "getHc_imag"
  PyObject* pFunc_getHc_imag = PyObject_GetAttrString(pModule, (char*)"getHc_imag");
  // Call the python function
  PyObject *pHc_imag;
  bool called_imag= false;
  if (pFunc_getHc_imag){
    if (PyCallable_Check(pFunc_getHc_imag)) {
      pHc_imag= PyObject_CallObject(pFunc_getHc_imag, NULL); // NULL: no input to getHc().
      PyErr_Print();
      called_imag= true;
    } else PyErr_Print();
  } else PyErr_Print();

  if (!called_imag || !PyList_Check(pHc_imag)) {
    if (mpirank_world == 0) printf("# No imag control Hamiltonian received. \n");
  } else {

    // Parse the result 
    // getHc() MUST return a python list of lists of lists of float elements for this to work:
    // for each oscillator k=0...Q-1: for each control term i=0...C^k-1: a list containing the flattened Hamiltonian Hc^k_i

    // Sanity check: the outer list should have Q == noscillators elements (each one being a list of Hamiltonians)
    int Q = 0;
    if (PyList_Check(pHc_imag)) {
      Q = PyList_Size(pHc_imag); 
      PyErr_Print();
    }
    if (Q != noscillators) {
     printf("Error parsing python function getHc_imag(): It should contain an (outer) list of length %d, but did return a list of length %d.\n", noscillators, Q);
     exit(1);
    }
    // printf("getHc(): Received an (outer) list of length %d\n", Q);

    // Receive and store Hamiltonian values
    std::vector<std::vector<double>> Hc_im_vals; // vector of vector of Hamiltonian values
    std::vector<std::vector<int>> Hc_im_ids;  // vector of vector of Hamiltonian id

    // Iterate over oscillators
    for (Py_ssize_t k=0; k<noscillators; k++){

      // Check number of control terms for this oscillator
      int ncontrol = 0;
      PyObject* pHck_imag= PyList_GetItem(pHc_imag,k); // Get list of Hamiltonians for oscillator k
      PyErr_Print();
      if (PyList_Check(pHck_imag)) { 
        ncontrol = PyList_Size(pHck_imag); 
        PyErr_Print();
      } else PyErr_Print();
      ncontrol_imag[k] = ncontrol;
      // printf("getHc(): For oscillator %d, received %d control Hamiltonians.\n", (int)k, ncontrol);

      // Iterate over control terms for this oscillator 
      for (Py_ssize_t i=0; i<ncontrol_imag[k]; i++){
        PyObject* pHcki_imag= PyList_GetItem(pHck_imag,i); // Get the Hamiltonian Hcki
        PyErr_Print();

        int Hcki_size_im = 0;
        if (PyList_Check(pHcki_imag)) {
          Hcki_size_im = PyList_Size(pHcki_imag);
          PyErr_Print();
        }
        int Hcki_size = Hcki_size_im;
        // printf("getHc(): Oscillator %d, term %i: Received a list of length = %d: ", (int)k,(int)i,Hcki_size);
        std::vector<double> Hcki_im_vals;
        std::vector<int> Hcki_im_ids;
        for (Py_ssize_t l=0; l<Hcki_size; l++){ // Iterate over elements
          PyObject* pval_im = PyList_GetItem(pHcki_imag,l);
          PyErr_Print();
          double Hcki_im_val = PyFloat_AsDouble(pval_im);
          PyErr_Print();
          if (fabs(Hcki_im_val) > 1e-14) {  // store only nonzeros
            Hcki_im_vals.push_back(Hcki_im_val);
            Hcki_im_ids.push_back(l);
          }
        }
        Hc_im_vals.push_back(Hcki_im_vals);
        Hc_im_ids.push_back(Hcki_im_ids);
      } // end of control term i for this oscillator k
    } // end of oscillator k

    /* Now place values into Ac_vec[k][l] for all oscillators k and all control terms l */

    int ioscil = 0;  // index for accessing Hamiltonian values inside Hc_im_vals/ids
    // Iterate over oscillators
    for (int k=0; k<noscillators; k++){

      // Iterate over control terms for this oscillator
      for (int i=0; i<ncontrol_imag[k]; i++){

        // Create a new control matrix
        Mat myAcMat_kl;
        Ac_vec[k].push_back(myAcMat_kl);

        MatCreate(PETSC_COMM_WORLD, &(Ac_vec[k][i]));
        MatSetType(Ac_vec[k][i], MATMPIAIJ);
        MatSetSizes(Ac_vec[k][i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
        MatSetUp(Ac_vec[k][i]);
        MatSetFromOptions(Ac_vec[k][i]);

        PetscInt ilow, iupp;
        MatGetOwnershipRange(Ac_vec[k][i], &ilow, &iupp);

        // Assemble -I_N \kron Hc^k_i + Hc^k_i \kron I_N (Lindblad) or -Hc^k_i (Schroedinger)
        // vals are in Hc_vals[ioscil][:]
        // Iterate over nonzero elements in Hc^k_i
        for (int l = 0; l<Hc_im_ids[ioscil].size(); l++) {
          // Get position in the Bc matrix
          int row = Hc_im_ids[ioscil][l] % sqdim;
          int col = Hc_im_ids[ioscil][l] / sqdim;

          if (lindbladtype == LindbladType::NONE){
            // Schroedinger
            // Assemble - B_c  
            double val = -1.*Hc_im_vals[ioscil][l];
            if (ilow <= row && row < iupp) MatSetValue(Ac_vec[k][i], row, col, val, ADD_VALUES);
          } else {
            // Lindblad
            // Assemble -I_N \kron B_c + B_c \kron I_N 
            for (int m=0; m<sqdim; m++){
              // first place all -v_ij in the -I_N\kron B_c term:
              int rowm = row + sqdim * m;
              int colm = col + sqdim * m;
              double val = -1.*Hc_im_vals[ioscil][l];
              if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][i], rowm, colm, val, ADD_VALUES);
              // Then add v_ij in the B_d^T \kron I_N term:
              rowm = col*sqdim + m;   // transpose!
              colm = row*sqdim + m;
              val = Hc_im_vals[ioscil][l];
              if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][i], rowm, colm, val, ADD_VALUES);
            }
          }
        } // end of elements in Hc^k_i
        ioscil++;
      } // end of i loop for control terms

      // Assemble the matrices for this oscillator
      for (int i=0; i<ncontrol_imag[k]; i++){
        MatAssemblyBegin(Ac_vec[k][i], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Ac_vec[k][i], MAT_FINAL_ASSEMBLY);
      }
    } // end of k loop for oscillators for IMAG Hamiltonian
  } // end of imag part reading from python

  // Clean up
  if (pFunc_getHc_real) Py_XDECREF(pFunc_getHc_real);  
  if (pFunc_getHc_imag) Py_XDECREF(pFunc_getHc_imag);  
#endif
}


void PythonInterface::receiveHcTransfer(int noscillators,std::vector<std::vector<TransferFunction*>>& transfer_Hc_re,std::vector<std::vector<TransferFunction*>>& transfer_Hc_im){
#ifdef WITH_PYTHON

  if (mpirank_world == 0) printf("Receiving transfer functions for control Hamiltonians Hc(t)...\n");

  /* First empty out the transfer_func vectors for each oscillator and create empty vectors of the correct size */
  for (int k=0; k<noscillators; k++){
    for (int icon = 0; icon < transfer_Hc_re[k].size(); icon++) delete transfer_Hc_re[k][icon];
    for (int icon = 0; icon < transfer_Hc_im[k].size(); icon++) delete transfer_Hc_im[k][icon];
    transfer_Hc_re[k].clear();
    transfer_Hc_im[k].clear();

    for (int icontrol = 0; icontrol<ncontrol_real[k]; icontrol++){
      transfer_Hc_re[k].push_back(NULL);
    }
    for (int icontrol = 0; icontrol<ncontrol_imag[k]; icontrol++){
      transfer_Hc_im[k].push_back(NULL);
    } 
  }


  /* Transfer function u(p(t)) for REAL part */

  // Get a reference to the required python functions
  PyObject *pFunc_getTr_real = pFunc_getTr_real = PyObject_GetAttrString(pModule, (char*)"getHcTransfer_real");  // transfer functions

  // Call the function
  PyObject* pTr_real;
  bool called_real=false;
  if (pFunc_getTr_real) {
    if (PyCallable_Check(pFunc_getTr_real)) {
      pTr_real = PyObject_CallObject(pFunc_getTr_real, NULL); // NULL: no input to getTransfer().
      PyErr_Print();
      called_real = true;
      if (PyList_Check(pTr_real)) assert(PyList_Size(pTr_real) == noscillators);
    } else PyErr_Print();
  } else PyErr_Print();

  // Iterate over oscillators
  for (Py_ssize_t k=0; k<noscillators; k++){

    if (!called_real || !PyList_Check(pTr_real)){
      // Default: If we can't find the python function "getTranfer_real", use identities for each control Hamiltonian term
      // printf("# Warning: Could not find transfer function 'getHcTransfer_real'. Using identity for oscillator %zd instead. \n",k);

      for (Py_ssize_t i=0; i<ncontrol_real[k]; i++){
        IdentityTransferFunction *transfer_ki =  new IdentityTransferFunction();
        transfer_Hc_re[k][i] = transfer_ki;
      }
    } else { 
      // If 'getHcTransfer_real' exists, get all transfer functions from python in terms of spline knots and coefs. 

      // Check number of given transfer functions for this oscillator 
      PyObject* pTrk_real = PyList_GetItem(pTr_real,k); // Get list of Function for oscillator k
      PyErr_Print();
      int Ck_r = 0;
      if (PyList_Check(pTrk_real)) { 
        Ck_r = PyList_Size(pTrk_real); 
        PyErr_Print();
      } else PyErr_Print();
      assert(Ck_r == ncontrol_real[k]);

      // Iterate over list for this oscillator 
      PyObject* pTrki, *pOrder_ki, *pknots_ki, *pcoefs_ki, *pTrki_knot, *pTrki_coef;
      for (Py_ssize_t i=0; i<ncontrol_real[k]; i++){
        pTrki = PyList_GetItem(pTrk_real,i); // Get spline for u^k_i(x). Returns [knots, coeffs, order]
        PyErr_Print();

        // Get knots, coeffs and order of the spline
        int uki_order=0;
        std::vector<double> uki_knots, uki_coefs;
        if (PyList_Check(pTrki)) {  // pTrki = [knots, coeffs, order]
          assert(PyList_Size(pTrki) == 3); 
          // Get order
          pOrder_ki = PyList_GetItem(pTrki,2); 
          uki_order = PyInt_AsLong(pOrder_ki);

          // Get knots coeffs and order for spline u^k_i(x)
          pknots_ki = PyList_GetItem(pTrki,0); 
          pcoefs_ki = PyList_GetItem(pTrki,1); 
          if (PyList_Check(pknots_ki) && PyList_Check(pcoefs_ki)) {  // pTrki = [knots, coeffs, order]
            for (Py_ssize_t l=0; l<PyList_Size(pknots_ki); l++) {
              pTrki_knot= PyList_GetItem(pknots_ki, l);
              double val = PyFloat_AsDouble(pTrki_knot);
              uki_knots.push_back( val );
              pTrki_coef= PyList_GetItem(pcoefs_ki, l);
              val = PyFloat_AsDouble(pTrki_coef);
              uki_coefs.push_back( val );
            }
          } else PyErr_Print();
        } else PyErr_Print();

        // Create the transfer spline for u^k_i and store it
        SplineTransferFunction *transfer_ki = new SplineTransferFunction(uki_order, uki_knots, uki_coefs);
        transfer_Hc_re[k][i] = transfer_ki;
      } // end of control term i for this oscillator k
    }
  } // end of oscillator k

  // Cleanup
  if (pFunc_getTr_real) Py_XDECREF(pFunc_getTr_real);  // pointer to getHc function



  /* Transfer function u(p(t)) for IMAGINARY part */

  // Get a reference to the required python functions
  PyObject *pFunc_getTr_imag = pFunc_getTr_imag= PyObject_GetAttrString(pModule, (char*)"getHcTransfer_imag");  // transfer functions

  // Call the functions
  PyObject* pTr_imag;
  bool called_imag=false;
  if (pFunc_getTr_imag) {
    if (PyCallable_Check(pFunc_getTr_imag)) {
      pTr_imag= PyObject_CallObject(pFunc_getTr_imag, NULL); // NULL: no input to getTransfer().
      PyErr_Print();
      called_imag= true;
      if (PyList_Check(pTr_imag)) assert(PyList_Size(pTr_imag) == noscillators);
    } else PyErr_Print();
  } else PyErr_Print();

  // Iterate over oscillators
  for (Py_ssize_t k=0; k<noscillators; k++){

    // Get number of given transfer functions for this oscillator
    if (!called_imag|| !PyList_Check(pTr_imag)){
      // Default: If we can't find the python function "getTranfer_imag", use identities for each control Hamiltonian term
      // printf("# Warning: Could not find transfer function 'getHcTransfer_imag'. Using identity for oscillator %zd instead. \n", k);
      for (Py_ssize_t i=0; i<ncontrol_imag[k]; i++){
        IdentityTransferFunction *transfer_ki =  new IdentityTransferFunction();
        transfer_Hc_im[k][i] = transfer_ki;
      }
    } else { 
      // If 'getHcTransfer_imag' exists, get all transfer functions from python in terms of spline knots and coefs. 

      // Check size of list for this oscillator
      PyObject* pTrk_imag= PyList_GetItem(pTr_imag,k); // Get list of Function for oscillator k
      PyErr_Print();
      int Ck_r = 0;
      if (PyList_Check(pTrk_imag)) { 
        Ck_r = PyList_Size(pTrk_imag); 
        PyErr_Print();
      } else PyErr_Print();
      assert(Ck_r == ncontrol_imag[k]);

      // Iterate over list for this oscillator 
      PyObject* pTrki, *pOrder_ki, *pknots_ki, *pcoefs_ki, *pTrki_knot, *pTrki_coef;
      for (Py_ssize_t i=0; i<ncontrol_imag[k]; i++){
        pTrki = PyList_GetItem(pTrk_imag,i); // Get spline for u^k_i(x). Returns [knots, coeffs, order]
        PyErr_Print();

        // Get knots, coeffs and order of the spline
        int uki_order=0;
        std::vector<double> uki_knots, uki_coefs;
        if (PyList_Check(pTrki)) {  // pTrki = [knots, coeffs, order]
          assert(PyList_Size(pTrki) == 3); 
          // Get order
          pOrder_ki = PyList_GetItem(pTrki,2); 
          uki_order = PyInt_AsLong(pOrder_ki);

          // Get knots coeffs and order for spline u^k_i(x)
          pknots_ki = PyList_GetItem(pTrki,0); 
          pcoefs_ki = PyList_GetItem(pTrki,1); 
          if (PyList_Check(pknots_ki) && PyList_Check(pcoefs_ki)) {  // pTrki = [knots, coeffs, order]
            for (Py_ssize_t l=0; l<PyList_Size(pknots_ki); l++) {
              pTrki_knot= PyList_GetItem(pknots_ki, l);
              double val = PyFloat_AsDouble(pTrki_knot);
              uki_knots.push_back( val );
              pTrki_coef= PyList_GetItem(pcoefs_ki, l);
              val = PyFloat_AsDouble(pTrki_coef);
              uki_coefs.push_back( val );
            }
          } else PyErr_Print();
        } else PyErr_Print();

        // Create the transfer spline for u^k_i and store it
        SplineTransferFunction *transfer_ki = new SplineTransferFunction(uki_order, uki_knots, uki_coefs);
        transfer_Hc_im[k][i] = transfer_ki;
      } // end of control term i for this oscillator k
    }
  } // end of oscillator k

  // Cleanup
  if (pFunc_getTr_imag) Py_XDECREF(pFunc_getTr_imag);  // pointer to getHc function

#endif
}



void PythonInterface::receiveHdt(std::vector<Mat>& Ad_vec, std::vector<Mat>& Bd_vec){
#ifdef WITH_PYTHON

  if (mpirank_world == 0) printf("Receiving time-dependent system Hamiltonians...\n");

  /* Reset Ad_vec and Bd_vec.  */
  for (int i= 0; i < Ad_vec.size(); i++) {
    if (Ad_vec[i] != NULL ) MatDestroy(&Ad_vec[i]);
  }
  Ad_vec.clear();
  for (int i= 0; i < Bd_vec.size(); i++) {
    if (Bd_vec[i] != NULL ) MatDestroy(&Bd_vec[i]);
  }
  Bd_vec.clear();

  int sqdim = dim_rho; //  N!
  int dim = dim_rho;
  if (lindbladtype !=LindbladType::NONE) dim = dim_rho*dim_rho;
  int nterms = 0;

  /* REAL part */

  // Receive and store Hamiltonian values and ids from python
  std::vector<std::vector<double>> Hdt_re_vals; // vector Hamiltonian values (one vector per coupling)
  std::vector<std::vector<int>> Hdt_re_ids;  // vector Hamiltonian id

  // Get a reference to the required python functions "getHdt_real"
  PyObject *pFunc_getHdt_real  = PyObject_GetAttrString(pModule, (char*)"getHdt_real");
  // Call the function
  PyObject* pHdt_real;
  bool called_real=false;
  if (pFunc_getHdt_real) {
    if (PyCallable_Check(pFunc_getHdt_real)) {
      pHdt_real = PyObject_CallObject(pFunc_getHdt_real, NULL); // NULL: no input
      PyErr_Print();
      called_real = true;
    } else PyErr_Print();
  } else PyErr_Print();

  if (!called_real || !PyList_Check(pHdt_real)){
    if (mpirank_world == 0) printf("# No time-dependent real Hamiltonian received. \n");
    nterms = 0;
    // If none given, leave the matrix empty.
  } else { 
    // Iterate over terms
    nterms = PyList_Size(pHdt_real);
    for (Py_ssize_t k=0; k<nterms; k++){

      // Get an item from the outer list (next Hamiltonian item)
      PyObject* pHdtk_real = PyList_GetItem(pHdt_real,k);
      PyErr_Print();
      int Hdtk_size= 0;
      if (PyList_Check(pHdtk_real)) {
        Hdtk_size = PyList_Size(pHdtk_real);
        PyErr_Print();
      }
      // printf("Hdtk_size=%d, sqdim=%d\n", Hdtk_size, sqdim);
      assert(Hdtk_size == sqdim*sqdim);

      // Iterate over the item and receive values and ids
      std::vector<double> Hdtk_vals;
      std::vector<int> Hdtk_ids;
      for (Py_ssize_t l=0; l<Hdtk_size; l++){ // Iterate over elements
        PyObject* pval_re = PyList_GetItem(pHdtk_real,l);
        double Hdtk_re_val = PyFloat_AsDouble(pval_re);
        PyErr_Print();
        if (fabs(Hdtk_re_val) > 1e-14) {  // store only nonzeros
          Hdtk_vals.push_back(Hdtk_re_val);
          Hdtk_ids.push_back(l);
        }
      } // end iterating over innter list
      Hdt_re_vals.push_back(Hdtk_vals);
      Hdt_re_ids.push_back(Hdtk_ids);
    } // end of out list of terms
    // Now we have Hdt_re_vals and Hdt_re_ids

    // // print out what we received from python 
    // for (int k=0; k<nterms; k++){
    //   printf("getHdt(): real(term %d): \n", k);
    //   for (int l=0; l < Hdt_re_vals[k].size(); l++){
    //     printf("(%d,%f) ", Hdt_re_ids[k][l], Hdt_re_vals[k][l]);
    //   }
    //   printf("\n");
    // }

    /* Now place values into Bd_vec */

    // Iterate over terms
    for (int k=0; k<nterms; k++){
      // First create the matrix 
      Mat myMat;
      Bd_vec.push_back(myMat);
      MatCreate(PETSC_COMM_WORLD, &(Bd_vec[k]));
      MatSetType(Bd_vec[k], MATMPIAIJ);
      MatSetSizes(Bd_vec[k], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetUp(Bd_vec[k]);
      MatSetFromOptions(Bd_vec[k]);

      PetscInt ilow, iupp;
      MatGetOwnershipRange(Bd_vec[k], &ilow, &iupp);

      // Iterate over elements in Hdtk
      for (int l = 0; l<Hdt_re_ids[k].size(); l++) {
        // Get position in the Bd matrix
        int row = Hdt_re_ids[k][l] % sqdim;
        int col = Hdt_re_ids[k][l] / sqdim;

        if (lindbladtype == LindbladType::NONE){
          // Schroedinger
          // Assemble - Bd  
          double val = -1.*Hdt_re_vals[k][l];
          if (ilow <= row && row < iupp) MatSetValue(Bd_vec[k], row, col, val, ADD_VALUES);
        } else {
          // Lindblad
          // Assemble -I_N \kron B_c + B_c \kron I_N 
          for (int m=0; m<sqdim; m++){
            // first place all -v_ij in the -I_N\kron B_c term:
            int rowm = row + sqdim * m;
            int colm = col + sqdim * m;
            double val = -1.*Hdt_re_vals[k][l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Bd_vec[k], rowm, colm, val, ADD_VALUES);
            // Then add v_ij in the B_d^T \kron I_N term:
            rowm = col*sqdim + m;   // transpose!
            colm = row*sqdim + m;
            val = Hdt_re_vals[k][l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Bd_vec[k], rowm, colm, val, ADD_VALUES);
          }
        }
      } // end of elements in Hdtk
      MatAssemblyBegin(Bd_vec[k], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Bd_vec[k], MAT_FINAL_ASSEMBLY);
    } // end of loop over terms
  } // End of setting the REAL valued Hamiltonian


  /* IMAGINARY part */

  // Receive and store Hamiltonian values and ids from python
  std::vector<std::vector<double>> Hdt_im_vals; // vector Hamiltonian values (one vector per coupling)
  std::vector<std::vector<int>> Hdt_im_ids;  // vector Hamiltonian id

  // Get a reference to the required python functions "getHdt_imag"
  PyObject *pFunc_getHdt_imag = PyObject_GetAttrString(pModule, (char*)"getHdt_imag");
  // Call the function
  PyObject* pHdt_imag;
  bool called_imag=false;
  if (pFunc_getHdt_imag) {
    if (PyCallable_Check(pFunc_getHdt_imag)) {
      pHdt_imag= PyObject_CallObject(pFunc_getHdt_imag, NULL); // NULL: no input
      PyErr_Print();
      called_imag= true;
      // Make sure the list contains <nterms> Hamiltonians
      if (PyList_Check(pHdt_imag)) assert(PyList_Size(pHdt_imag) == nterms);
    } else PyErr_Print();
  } else PyErr_Print();
  // Now we have pHdt_imag: [ H01, H02,... ,H12, H13,... ] length = nterms = Q*(Q-1)/2

  if (!called_imag|| !PyList_Check(pHdt_imag)){
    if (mpirank_world == 0) printf("# No time-dependent imag Hamiltonian received. \n");
    // If none given, leave the matrix empty. 
  } else { 
    // Iterate over terms
    for (Py_ssize_t k=0; k<nterms; k++){

      // Get an item from the outer list (next Hamiltonian item)
      PyObject* pHdtk_imag = PyList_GetItem(pHdt_imag,k);
      PyErr_Print();
      int Hdtk_size= 0;
      if (PyList_Check(pHdtk_imag)) {
        Hdtk_size = PyList_Size(pHdtk_imag);
        PyErr_Print();
      }
      assert(Hdtk_size == sqdim*sqdim);

      // Iterate over the item and receive values and ids
      std::vector<double> Hdtk_vals;
      std::vector<int> Hdtk_ids;
      for (Py_ssize_t l=0; l<Hdtk_size; l++){ // Iterate over elements
        PyObject* pval_im = PyList_GetItem(pHdtk_imag,l);
        double Hdtk_im_val = PyFloat_AsDouble(pval_im);
        PyErr_Print();
        if (fabs(Hdtk_im_val) > 1e-14) {  // store only nonzeros
          Hdtk_vals.push_back(Hdtk_im_val);
          Hdtk_ids.push_back(l);
        }
      } // end iterating over innter list
      Hdt_im_vals.push_back(Hdtk_vals);
      Hdt_im_ids.push_back(Hdtk_ids);
    } // end of out list of terms
    // Now we have Hdt_im_vals and Hdt_im_ids

    // // print out what we received from python 
    // for (int k=0; k<nterms; k++){
    //   printf("getHdt(): imag(term %d): \n", k);
    //   for (int l=0; l < Hdt_im_vals[k].size(); l++){
    //     printf("(%d,%f) ", Hdt_im_ids[k][l], Hdt_im_vals[k][l]);
    //   }
    //   printf("\n");
    // }

    /* Now place values into Ad_vec */

    // Iterate over terms
    for (int k=0; k<nterms; k++){
      // First create the matrix 
      Mat myMat;
      Ad_vec.push_back(myMat);
      MatCreate(PETSC_COMM_WORLD, &(Ad_vec[k]));
      MatSetType(Ad_vec[k], MATMPIAIJ);
      MatSetSizes(Ad_vec[k], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetUp(Ad_vec[k]);
      MatSetFromOptions(Ad_vec[k]);

      PetscInt ilow, iupp;
      MatGetOwnershipRange(Ad_vec[k], &ilow, &iupp);

      // Iterate over elements in Hdtk
      for (int l = 0; l<Hdt_im_ids[k].size(); l++) {
        // Get position in the Bd matrix
        int row = Hdt_im_ids[k][l] % sqdim;
        int col = Hdt_im_ids[k][l] / sqdim;

        if (lindbladtype == LindbladType::NONE){
          // Schroedinger
          // Assemble Ad
          double val = Hdt_im_vals[k][l];
          if (ilow <= row && row < iupp) MatSetValue(Ad_vec[k], row, col, val, ADD_VALUES);
        } else {
          // Lindblad
          // Assemble I_N \kron A_d - A_d^T \kron I_N 
          for (int m=0; m<sqdim; m++){
            // first place all v_ij in the I_N\kron A_d term:
            int rowm = row + sqdim * m;
            int colm = col + sqdim * m;
            double val = Hdt_im_vals[k][l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Ad_vec[k], rowm, colm, val, ADD_VALUES);
            // Then add -v_ij in the -A_d^T \kron I_N term:
            rowm = col*sqdim + m;   // transpose!
            colm = row*sqdim + m;
            val = -1.*Hdt_im_vals[k][l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Ad_vec[k], rowm, colm, val, ADD_VALUES);
          }
        }
      } // end of elements in Hdtk
      MatAssemblyBegin(Ad_vec[k], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Ad_vec[k], MAT_FINAL_ASSEMBLY);
    } // end of loop over terms
  } // End of setting the IMAG valued Hamiltonian

  // Clean up
  if (pFunc_getHdt_real) Py_XDECREF(pFunc_getHdt_real);  
  if (pFunc_getHdt_imag) Py_XDECREF(pFunc_getHdt_imag);  
#endif
}


void PythonInterface::receiveHdtTransfer(int nterms, std::vector<TransferFunction*>& transfer_Hdt_re, std::vector<TransferFunction*>& transfer_Hdt_im){
#ifdef WITH_PYTHON


 if (mpirank_world == 0) printf("Receiving transfer functions for time-dependent system Hamiltonians Hd(t)...\n");

  /* First empty out the transfer_func vectors for each oscillator and create empty vectors of the correct size */
  transfer_Hdt_re.clear();
  transfer_Hdt_im.clear();

  /* Transfer function u(t) for REAL part: u(t)*Real(Hdt) */

  // Get a reference to the required python functions
  PyObject *pFunc_getTr_real = pFunc_getTr_real = PyObject_GetAttrString(pModule, (char*)"getHdtTransfer_real");  // transfer functions

  // Call the function if it exists
  PyObject* pTr_real;
  bool called_real=false;
  if (pFunc_getTr_real) {
    if (PyCallable_Check(pFunc_getTr_real)) {
      pTr_real = PyObject_CallObject(pFunc_getTr_real, NULL); // NULL: no input to getTransfer().
      PyErr_Print();
      called_real = true;
      if (PyList_Check(pTr_real)) assert(PyList_Size(pTr_real) >= nterms);
    } else PyErr_Print();
  } else PyErr_Print();

  // Iterate over Hamiltonian terms
  for (Py_ssize_t k=0; k<nterms; k++){

    // Get number of given transfer functions for this oscillator
    if (!called_real || !PyList_Check(pTr_real)){
      // Default: If we can't find the python function "getTranfer_real", use identity
      // printf("# Warning: Could not find transfer function 'getHdtTransfer_real'. Using constant u=1.0 instead. \n");
      ConstantTransferFunction *transfer_k =  new ConstantTransferFunction(1.0);
      transfer_Hdt_re.push_back(transfer_k);
    } else { 
      // If 'getHdtTransfer_real' exists, get all transfer functions from python in terms of spline knots and coefs. 

      // Check size of list for this term
      PyObject* pTrk= PyList_GetItem(pTr_real,k); // Get list [knots, coeffs, order]
      PyErr_Print();
      int Ck_r = 0;
      if (PyList_Check(pTrk)) { 
        Ck_r = PyList_Size(pTrk); 
        PyErr_Print();
      } else PyErr_Print();
      assert(Ck_r == 3);

      // Get order
      PyObject* pOrder_k = PyList_GetItem(pTrk,2); 
      int uk_order = PyInt_AsLong(pOrder_k);

      // Get knots coeffs and order
      std::vector<double> uk_knots;
      std::vector<double> uk_coefs;
      PyObject* pknots_k = PyList_GetItem(pTrk,0); 
      PyObject* pcoefs_k = PyList_GetItem(pTrk,1); 
      if (PyList_Check(pknots_k) && PyList_Check(pcoefs_k)) {  
        PyObject* pTrk_knot, *pTrk_coef;
        for (Py_ssize_t l=0; l<PyList_Size(pknots_k); l++) {
              pTrk_knot= PyList_GetItem(pknots_k, l);
              double val = PyFloat_AsDouble(pTrk_knot);
              uk_knots.push_back( val );
              pTrk_coef= PyList_GetItem(pcoefs_k, l);
              val = PyFloat_AsDouble(pTrk_coef);
              uk_coefs.push_back( val );
        }
      } else PyErr_Print();

      // Create the transfer spline for u^k and store it
      SplineTransferFunction *transfer_k = new SplineTransferFunction(uk_order, uk_knots, uk_coefs);
      transfer_Hdt_re.push_back(transfer_k);
    }
  } // end of term k

  // Cleanup
  if (pFunc_getTr_real) Py_XDECREF(pFunc_getTr_real);  // pointer to getHc function
  
  /* Transfer function v(t) for IMAGINARY part: v(t)*Imag(Hdt) */

  // Get a reference to the required python functions
  PyObject *pFunc_getTr_imag = pFunc_getTr_imag = PyObject_GetAttrString(pModule, (char*)"getHdtTransfer_imag");  // transfer functions

  // Call the function if it exists
  PyObject* pTr_imag;
  bool called_imag=false;
  if (pFunc_getTr_imag) {
    if (PyCallable_Check(pFunc_getTr_imag)) {
      pTr_imag = PyObject_CallObject(pFunc_getTr_imag, NULL); // NULL: no input to getTransfer().
      PyErr_Print();
      called_imag = true;
      if (PyList_Check(pTr_imag)) assert(PyList_Size(pTr_imag) >= nterms);
    } else PyErr_Print();
  } else PyErr_Print();

  // Iterate over Hamiltonian terms
  for (Py_ssize_t k=0; k<nterms; k++){

    // Get number of given transfer functions for this oscillator
    if (!called_imag || !PyList_Check(pTr_imag)){
      // Default: If we can't find the python function "getTranfer_imag", use identity
      // printf("# Warning: Could not find transfer function 'getHdtTransfer_imag'. Using constant u=1.0 instead. \n");
      ConstantTransferFunction *transfer_k =  new ConstantTransferFunction(1.0);
      transfer_Hdt_im.push_back( transfer_k);
    } else { 
      // If 'getHdtTransfer_imag' exists, get all transfer functions from python in terms of spline knots and coefs. 

      // Check size of list for this term
      PyObject* pTrk= PyList_GetItem(pTr_imag,k); // Get list [knots, coeffs, order]
      PyErr_Print();
      int Ck_r = 0;
      if (PyList_Check(pTrk)) { 
        Ck_r = PyList_Size(pTrk); 
        PyErr_Print();
      } else PyErr_Print();
      assert(Ck_r == 3);

      // Get order
      PyObject* pOrder_k = PyList_GetItem(pTrk,2); 
      int uk_order = PyInt_AsLong(pOrder_k);

      // Get knots coeffs and order
      std::vector<double> uk_knots;
      std::vector<double> uk_coefs;
      PyObject* pknots_k = PyList_GetItem(pTrk,0); 
      PyObject* pcoefs_k = PyList_GetItem(pTrk,1); 
      if (PyList_Check(pknots_k) && PyList_Check(pcoefs_k)) {  
        PyObject* pTrk_knot, *pTrk_coef;
        for (Py_ssize_t l=0; l<PyList_Size(pknots_k); l++) {
              pTrk_knot= PyList_GetItem(pknots_k, l);
              double val = PyFloat_AsDouble(pTrk_knot);
              uk_knots.push_back( val );
              pTrk_coef= PyList_GetItem(pcoefs_k, l);
              val = PyFloat_AsDouble(pTrk_coef);
              uk_coefs.push_back( val );
        }
      } else PyErr_Print();

      // Create the transfer spline for u^k_i and store it
      SplineTransferFunction *transfer_k = new SplineTransferFunction(uk_order, uk_knots, uk_coefs);
      transfer_Hdt_im.push_back(transfer_k);
    }
  } // end of term k

  // Cleanup
  if (pFunc_getTr_imag) Py_XDECREF(pFunc_getTr_imag);  // pointer to getHc function



#endif
}