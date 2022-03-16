#include "pythoninterface.hpp"

PythonInterface::PythonInterface(){
}


PythonInterface::PythonInterface(std::string python_file, LindbladType lindbladtype_) {

  lindbladtype = lindbladtype_;

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

void PythonInterface::receiveHd(Mat& Bd){
#ifdef WITH_PYTHON

  printf("Receiving system Hamiltonian...\n");

  // Get a pointer to the python function "getHd"
  PyObject* pFunc_getHd = PyObject_GetAttrString(pModule, (char*)"getHd"); 

  // Call the python function.
  PyObject *pHd;
  if (pFunc_getHd && PyCallable_Check(pFunc_getHd)) {
    pHd = PyObject_CallObject(pFunc_getHd, NULL); // NULL: no input to getHd().
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
  Py_XDECREF(pFunc_getHd); 

  /* Get matrix size */ 
  PetscInt dim = 0;
  MatGetSize(Bd, &dim, NULL); // could be N^2 or N

  /* Write values into sparse Petsc matrix Bd. */
  // Bd had been allocated before. Destroy and reallocate it here. 
  MatDestroy(&Bd);
  MatCreate(PETSC_COMM_WORLD, &Bd);
  MatSetType(Bd, MATMPIAIJ);
  MatSetSizes(Bd, PETSC_DECIDE, PETSC_DECIDE, dim, dim); // dim = N^2 for Lindblad, N for Schroedinger
  // MatMPIAIJSetPreallocation(Bd, 1, NULL, 1, NULL); // TODO: Should preallocate Bd?
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
#endif
}



void PythonInterface::receiveHc(int noscillators, Mat** Ac_vec, Mat** Bc_vec, std::vector<int>& ncontrolterms){
#ifdef WITH_PYTHON

  printf("Receiving control Hamiltonian terms...\n");

  // Get a reference to the python function "getHc_real" and "getHc_imag"
  PyObject* pFunc_getHc_real = PyObject_GetAttrString(pModule, (char*)"getHc_real");
  PyObject* pFunc_getHc_imag = PyObject_GetAttrString(pModule, (char*)"getHc_imag");

  // Call the python functions
  PyObject *pHc_real, *pHc_imag;
  if (pFunc_getHc_real && PyCallable_Check(pFunc_getHc_real)) {
    pHc_real = PyObject_CallObject(pFunc_getHc_real, NULL); // NULL: no input to getHc().
    PyErr_Print();
  } else PyErr_Print();
  if (pFunc_getHc_imag && PyCallable_Check(pFunc_getHc_imag)) {
    pHc_imag = PyObject_CallObject(pFunc_getHc_imag, NULL); // NULL: no input to getHc().
    PyErr_Print();
  } else PyErr_Print();

  // Parse the result 
  // getHc() MUST return a python list of lists of lists of float elements for this to work:
  // for each oscillator k=0...Q-1: for each control term i=0...C^k-1: a list containing the flattened Hamiltonian Hc^k_i
  int Q_r = 0;
  if (PyList_Check(pHc_real)) {
    Q_r = PyList_Size(pHc_real); 
    PyErr_Print();
  }
  int Q_i = 0;
  if (PyList_Check(pHc_imag)) {
    Q_i = PyList_Size(pHc_imag); 
    PyErr_Print();
  }
  assert(Q_r == Q_i);
  int Q = Q_r;
  // Sanity check: the outer loop should have Q == noscillators elements (each one being a list of Hamiltonians)
  if (Q != noscillators) {
    printf("Error parsing python function getHc(): It should contain an (outer) list of length %d, but did return a list of length %d.\n", noscillators, Q);
    exit(1);
  }
  // printf("getHc(): Received an (outer) list of length %d\n", Q);

  std::vector<std::vector<double>> Hc_re_vals; // vector of vector of Hamiltonian values
  std::vector<std::vector<double>> Hc_im_vals; // vector of vector of Hamiltonian values
  std::vector<std::vector<int>> Hc_re_ids;  // vector of vector of Hamiltonian id
  std::vector<std::vector<int>> Hc_im_ids;  // vector of vector of Hamiltonian id

  // Iterate over oscillators
  for (Py_ssize_t k=0; k<Q; k++){

    // Check number of control terms for this oscillator
    int Ck_r = 0;
    int Ck_i = 0;
    PyObject* pHck_real = PyList_GetItem(pHc_real,k); // Get list of Hamiltonians for oscillator k
    PyObject* pHck_imag = PyList_GetItem(pHc_imag,k); // Get list of Hamiltonians for oscillator k
    PyErr_Print();
    if (PyList_Check(pHck_real)) { 
      Ck_r = PyList_Size(pHck_real); 
      PyErr_Print();
    } else PyErr_Print();
    if (PyList_Check(pHck_imag)) { 
      Ck_i = PyList_Size(pHck_imag); 
      PyErr_Print();
    } else PyErr_Print();
    assert(Ck_i == Ck_r);
    ncontrolterms[k] = Ck_r;
    // printf("getHc(): For oscillator %d, received %d control Hamiltonians.\n", (int)k, ncontrolterms[k]);

    // Iterate over control terms for this oscillator 
    for (Py_ssize_t i=0; i<ncontrolterms[k]; i++){
      PyObject* pHcki_real = PyList_GetItem(pHck_real,i); // Get the Hamiltonian Hcki
      PyObject* pHcki_imag = PyList_GetItem(pHck_imag,i); // Get the Hamiltonian Hcki
      PyErr_Print();

      int Hcki_size_re = 0;
      int Hcki_size_im = 0;
      if (PyList_Check(pHcki_real)) {
        Hcki_size_re = PyList_Size(pHcki_real);
        PyErr_Print();
      }
      if (PyList_Check(pHcki_imag)) {
        Hcki_size_im = PyList_Size(pHcki_imag);
        PyErr_Print();
      }
      assert(Hcki_size_re == Hcki_size_im);
      int Hcki_size = Hcki_size_re;
      // printf("getHc(): Oscillator %d, term %i: Received a list of length = %d: ", (int)k,(int)i,Hcki_size);
      std::vector<double> Hcki_re_vals;
      std::vector<double> Hcki_im_vals;
      std::vector<int> Hcki_re_ids;
      std::vector<int> Hcki_im_ids;
      for (Py_ssize_t l=0; l<Hcki_size; l++){ // Iterate over elements
        PyObject* pval_re = PyList_GetItem(pHcki_real,l);
        PyObject* pval_im = PyList_GetItem(pHcki_imag,l);
        PyErr_Print();
        double Hcki_re_val = PyFloat_AsDouble(pval_re);
        double Hcki_im_val = PyFloat_AsDouble(pval_im);
        PyErr_Print();
        if (fabs(Hcki_re_val) > 1e-14) {  // store only nonzeros
          Hcki_re_vals.push_back(Hcki_re_val);
          Hcki_re_ids.push_back(l);
        }
        if (fabs(Hcki_im_val) > 1e-14) {  // store only nonzeros
          Hcki_im_vals.push_back(Hcki_im_val);
          Hcki_im_ids.push_back(l);
        }
      }
      Hc_re_vals.push_back(Hcki_re_vals);
      Hc_re_ids.push_back(Hcki_re_ids);
      Hc_im_vals.push_back(Hcki_im_vals);
      Hc_im_ids.push_back(Hcki_im_ids);
    } // end of control term i for this oscillator k
  } // end of oscillator k

  // // print out what we received from python 
  // int id = 0;
  // for (int k=0; k<noscillators; k++){
  //   printf("getHc(): Oscillator %d: %d control terms: \n", k, ncontrolterms[k]);
  //   for (int i=0; i<ncontrolterms[k]; i++){
  //     printf("  %dth control: \n", i);
  //     for (int l=0; l < Hc_re_vals[id].size(); l++){
  //       printf("(%d,%f) ", Hc_re_ids[id][l], Hc_re_vals[id][l]);
  //     }
  //     printf("\n");
  //     printf("  + im * \n");
  //     for (int l=0; l < Hc_im_vals[id].size(); l++){
  //       printf("(%d,%f) ", Hc_im_ids[id][l], Hc_im_vals[id][l]);
  //     }
  //     id++;
  //     printf("\n");
  //   }
  // }

  // Clean up
  Py_XDECREF(pFunc_getHc_real);  // pointer to getHc function
  Py_XDECREF(pFunc_getHc_imag);  // pointer to getHc function


  // Store number of control terms
  ncontrolterms_store = ncontrolterms;


  /* Write control Hamiltonians into sparse matrices -I\kron Hc + Hc \kron I (if Lindblad), or -Hc if Schroedinger */
  int ioscil = 0;
  for (int k=0; k<noscillators; k++){

    /* Get matrix size */ 
    PetscInt dim = 0;
    MatGetSize(Ac_vec[k][0], &dim, NULL); // could be N^2 or N
    PetscInt ilow, iupp;
    MatGetOwnershipRange(Ac_vec[k][0], &ilow, &iupp);

    // The first one has been allocated for default sparse mat setting, so need to destroy first. 
    MatDestroy(&(Ac_vec[k][0])); 
    MatDestroy(&(Bc_vec[k][0]));
    delete [] Ac_vec[k];
    delete [] Bc_vec[k];

    // Create new mats for this oscillator, minimum one. If ncontrolterms==0, we still create one but it will be empty. // Why? TODO.
    int nHams = std::max(ncontrolterms[k], 1);
    printf("Creating %d control Mats for oscillator %d\n", nHams, k);
    Ac_vec[k] = new Mat[nHams];
    Bc_vec[k] = new Mat[nHams];
    // Always create the first Hamiltonian (might be empty.)
    MatCreate(PETSC_COMM_WORLD, &(Ac_vec[k][0]));
    MatCreate(PETSC_COMM_WORLD, &(Bc_vec[k][0]));
    MatSetType(Ac_vec[k][0], MATMPIAIJ);
    MatSetType(Bc_vec[k][0], MATMPIAIJ);
    MatSetSizes(Ac_vec[k][0], PETSC_DECIDE, PETSC_DECIDE, dim, dim); // dim = N^2 for Lindblad, dim=N for Schroedinger
    MatSetSizes(Bc_vec[k][0], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatSetUp(Ac_vec[k][0]);
    MatSetUp(Bc_vec[k][0]);
    MatSetFromOptions(Ac_vec[k][0]);
    MatSetFromOptions(Bc_vec[k][0]);
    // Create remaining matrices for this oscillator
    for (int i=1; i<ncontrolterms[k]; i++){
      MatCreate(PETSC_COMM_WORLD, &(Bc_vec[k][i]));
      MatCreate(PETSC_COMM_WORLD, &(Ac_vec[k][i]));
      MatSetType(Bc_vec[k][i], MATMPIAIJ);
      MatSetType(Ac_vec[k][i], MATMPIAIJ);
      MatSetSizes(Bc_vec[k][i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetSizes(Ac_vec[k][i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetUp(Bc_vec[k][i]);
      MatSetUp(Ac_vec[k][i]);
      MatSetFromOptions(Bc_vec[k][i]);
      MatSetFromOptions(Ac_vec[k][i]);
    }


    int sqdim = dim; // could be N^2 or N
    if (lindbladtype != LindbladType::NONE) sqdim = (int) sqrt(dim); // sqdim = N 

    // Set values for each control terms for this oscillator
    for (int i=0; i<ncontrolterms[k]; i++){
      // Assemble -I_N \kron Hc^k_i + Hc^k_i \kron I_N (Lindblad) or -Hc^k_i (Schroedinger)
      // vals are in Hc_vals[ioscil][:]
      /* REAL part */
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
      /* IMAGINARY part */
      // Iterate over nonzero elements in Hc^k_i
      for (int l = 0; l<Hc_im_ids[ioscil].size(); l++) {
        // Get position in the Ac matrix
        int row = Hc_im_ids[ioscil][l] % sqdim;
        int col = Hc_im_ids[ioscil][l] / sqdim;

        if (lindbladtype == LindbladType::NONE){
          // Schroedinger
          // Assemble A_c  
          double val = Hc_im_vals[ioscil][l];
          if (ilow <= row && row < iupp) MatSetValue(Ac_vec[k][i], row, col, val, ADD_VALUES);
        } else {
          // Lindblad
          // Assemble I_N \kron A_c - A_c^T \kron I_N 
          for (int m=0; m<sqdim; m++){
            // first place all v_ij in the I_N\kron A_c term:
            int rowm = row + sqdim * m;
            int colm = col + sqdim * m;
            double val = Hc_im_vals[ioscil][l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][i], rowm, colm, val, ADD_VALUES);
            // Then add -v_ij in the -A_c^T \kron I_N term:
            rowm = col*sqdim + m; // transpose !
            colm = row*sqdim + m;
            val = -1.*Hc_im_vals[ioscil][l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][i], rowm, colm, val, ADD_VALUES);
          }
        }
      }
      ioscil++;
    } // end of i loop for control terms

    // Always assemble the first matrix for this oscillator (might be empty)
    MatAssemblyBegin(Ac_vec[k][0], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ac_vec[k][0], MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Bc_vec[k][0], MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Bc_vec[k][0], MAT_FINAL_ASSEMBLY);
    // Assemble the other ones for this oscillator
    for (int i=1; i<ncontrolterms[k]; i++){
      MatAssemblyBegin(Bc_vec[k][i], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Bc_vec[k][i], MAT_FINAL_ASSEMBLY);
      MatAssemblyBegin(Ac_vec[k][i], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Ac_vec[k][i], MAT_FINAL_ASSEMBLY);
    }
  } // end of k loop for oscillators

#endif
}


void PythonInterface::receiveTransfer(int noscillators,std::vector<std::vector<fitpackpp::BSplineCurve*>>& transfer_func_re, std::vector<std::vector<fitpackpp::BSplineCurve*>>& transfer_func_im){
#ifdef WITH_PYTHON

  printf("Receiving transfer functions...\n");

  // Get a reference to the required python functions
  PyObject *pFunc_getTr_real = PyObject_GetAttrString(pModule, (char*)"getTransfer_real");  // transfer functions
  PyObject *pFunc_getTr_imag = PyObject_GetAttrString(pModule, (char*)"getTransfer_imag");  // transfer functions


  // Call the functions 
  PyObject* pTr_real, *pTr_imag;
  if (pFunc_getTr_real && PyCallable_Check(pFunc_getTr_real)) {
    pTr_real = PyObject_CallObject(pFunc_getTr_real, NULL); // NULL: no input to getTransfer().
    PyErr_Print();
  } else PyErr_Print();
  if (pFunc_getTr_imag && PyCallable_Check(pFunc_getTr_imag)) {
    pTr_imag = PyObject_CallObject(pFunc_getTr_imag, NULL); // NULL: no input to getTransfer().
    PyErr_Print();
  } else PyErr_Print();

  // getTransfer() MUST return a python list of lists of [splines knots, and coeffs, and order]:
  // for each oscillator k=0...Q-1: for each control term i=0...C^k-1: one transfer function u^k_i(x) (real-valued) given in terms of spline knots (list), coefficients(list) and order (int)
  int Q_r = 0;
  int Q_i = 0;
  if (PyList_Check(pTr_real)) {
    Q_r = PyList_Size(pTr_real); 
    PyErr_Print();
  }
  if (PyList_Check(pTr_imag)) {
    Q_i = PyList_Size(pTr_imag); 
    PyErr_Print();
  }
  assert(Q_r == Q_i);
  int Q = Q_r;
  // Sanity check 
  if (Q != noscillators) {
    printf("Error parsing python function getTr(): It should contain an (outer) list of length %d, but did return a list of length %d.\n", noscillators, Q);
    exit(1);
  }

  // Iterate over oscillators
  for (Py_ssize_t k=0; k<Q; k++){
    // Check number of control terms for this oscillator
    int Ck_r = 0;
    PyObject* pTrk_real = PyList_GetItem(pTr_real,k); // Get list of Function for oscillator k
    PyErr_Print();
    if (PyList_Check(pTrk_real)) { 
      Ck_r = PyList_Size(pTrk_real); 
      PyErr_Print();
    } else PyErr_Print();
    int Ck_i = 0;
    PyObject* pTrk_imag = PyList_GetItem(pTr_imag,k); // Get list of Function for oscillator k
    PyErr_Print();
    if (PyList_Check(pTrk_imag)) { 
      Ck_i = PyList_Size(pTrk_imag); 
      PyErr_Print();
    } else PyErr_Print();
    // printf("getTr(): For oscillator %d, received %d transfer functions.\n", (int)k, ncontrolterms[k]);
    int Ck = Ck_r;
    assert(Ck_r == Ck_i);
    assert(Ck == ncontrolterms_store[k]);

    // Iterate over control terms for this oscillator 
    std::vector<fitpackpp::BSplineCurve*> transfer_k_re;
    std::vector<fitpackpp::BSplineCurve*> transfer_k_im;
    PyObject* pTrki_real, *pTrki_imag, *pOrder_ki_real,* pOrder_ki_imag, *pknots_ki_real,*pknots_ki_imag, *pcoefs_ki_real, *pcoefs_ki_imag, *pTrki_knot_real, *pTrki_knot_imag, *pTrki_coef_real,*pTrki_coef_imag;
    for (Py_ssize_t i=0; i<ncontrolterms_store[k]; i++){
      pTrki_real = PyList_GetItem(pTrk_real,i); // Get spline for u^k_i(x). Returns [knots, coeffs, order]
      pTrki_imag = PyList_GetItem(pTrk_imag,i); // Get spline for u^k_i(x). Returns [knots, coeffs, order]
      PyErr_Print();

      std::vector<double> uki_re_knots, uki_re_coefs;
      std::vector<double> uki_im_knots, uki_im_coefs;
      int uki_re_order=0;
      int uki_im_order=0;
      if (PyList_Check(pTrki_real)) {  // pTrki = [knots, coeffs, order]
        assert(PyList_Size(pTrki_real) == 3); 
        // Get order
        pOrder_ki_real = PyList_GetItem(pTrki_real,2); 
        pOrder_ki_imag = PyList_GetItem(pTrki_imag,2); 
        uki_re_order = PyInt_AsLong(pOrder_ki_real);
        uki_im_order = PyInt_AsLong(pOrder_ki_imag);

        // Get knots coeffs and order for spline u^k_i(x)
        pknots_ki_real = PyList_GetItem(pTrki_real,0); 
        pknots_ki_imag = PyList_GetItem(pTrki_imag,0); 
        pcoefs_ki_real = PyList_GetItem(pTrki_real,1); 
        pcoefs_ki_imag = PyList_GetItem(pTrki_imag,1); 
        if (PyList_Check(pknots_ki_real) && PyList_Check(pcoefs_ki_real)) {  // pTrki = [knots, coeffs, order]
          for (Py_ssize_t l=0; l<PyList_Size(pknots_ki_real); l++) {
            pTrki_knot_real = PyList_GetItem(pknots_ki_real, l);
            double val = PyFloat_AsDouble(pTrki_knot_real);
            uki_re_knots.push_back( val );
            pTrki_coef_real = PyList_GetItem(pcoefs_ki_real, l);
            val = PyFloat_AsDouble(pTrki_coef_real);
            uki_re_coefs.push_back( val );
          }
        } else PyErr_Print();
        if (PyList_Check(pknots_ki_imag) && PyList_Check(pcoefs_ki_imag)) {  // pTrki = [knots, coeffs, order]
          for (Py_ssize_t l=0; l<PyList_Size(pknots_ki_imag); l++) {
            pTrki_knot_imag = PyList_GetItem(pknots_ki_imag, l);
            double val = PyFloat_AsDouble(pTrki_knot_imag);
            uki_im_knots.push_back( val );
            pTrki_coef_imag = PyList_GetItem(pcoefs_ki_imag, l);
            val = PyFloat_AsDouble(pTrki_coef_imag);
            uki_im_coefs.push_back( val );
          }
        } else PyErr_Print();

      } else PyErr_Print();

      // Create the spline for u^k_i and store it
      fitpackpp::BSplineCurve* transfer_ki_real = new fitpackpp::BSplineCurve(uki_re_knots, uki_re_coefs, uki_re_order);
      fitpackpp::BSplineCurve* transfer_ki_imag = new fitpackpp::BSplineCurve(uki_im_knots, uki_im_coefs, uki_im_order);
      transfer_k_re.push_back(transfer_ki_real);
      transfer_k_im.push_back(transfer_ki_imag);
    } // end of control term i for this oscillator k
      
    // Store the transfer splines for each oscillator in the master equation
    transfer_func_re.push_back(transfer_k_re);
    transfer_func_im.push_back(transfer_k_im);
  } // end of oscillator k

  // Cleanup
  Py_XDECREF(pFunc_getTr_real);  // pointer to getHc function
  Py_XDECREF(pFunc_getTr_imag);  // pointer to getHc function

#endif
}



void PythonInterface::receiveHdt(int noscillators, Mat* Ad_vec, Mat* Bd_vec){
#ifdef WITH_PYTHON

  printf("Receiving time-dependent system Hamiltonians...\n");

  /* Reset Ad_vec and Bd_vec. TODO: Fill in from python. */
  for (int i= 0; i < noscillators*(noscillators-1)/2; i++) {
    if (Ad_vec[i] != NULL )  {
      int dim = 0;
      MatGetSize(Ad_vec[i], &dim, NULL);

      MatDestroy(&Ad_vec[i]);
      MatDestroy(&Bd_vec[i]);


      MatCreate(PETSC_COMM_WORLD, &Ad_vec[i]);
      MatCreate(PETSC_COMM_WORLD, &Bd_vec[i]);
      MatSetType(Ad_vec[i], MATMPIAIJ);
      MatSetType(Bd_vec[i], MATMPIAIJ);
      MatSetSizes(Ad_vec[i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetSizes(Bd_vec[i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetUp(Ad_vec[i]);
      MatSetUp(Bd_vec[i]);
      MatSetFromOptions(Ad_vec[i]);
      MatSetFromOptions(Bd_vec[i]);
      
      MatAssemblyBegin(Ad_vec[i], MAT_FINAL_ASSEMBLY);
      MatAssemblyBegin(Bd_vec[i], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Ad_vec[i], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Ad_vec[i], MAT_FINAL_ASSEMBLY);
    }
  }

#endif
}