#include "pythoninterface.hpp"

PythonInterface::PythonInterface(){
}


PythonInterface::PythonInterface(std::string hamiltonian_file_, LindbladType lindbladtype_, int dim_rho_, bool quietmode_) {

  lindbladtype = lindbladtype_;
  dim_rho = dim_rho_;
  hamiltonian_file = hamiltonian_file_;
  quietmode=quietmode_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
}

PythonInterface::~PythonInterface(){
}

void PythonInterface::receiveHsys(Mat& Bd){
  PetscInt ilow, iupp;
  MatGetOwnershipRange(Bd, &ilow, &iupp);

  /* Get sizes */ 
  PetscInt dim = 0;
  MatGetSize(Bd, &dim, NULL); // could be N^2 or N
  int sqdim = dim;
  if (lindbladtype != LindbladType::NONE) sqdim = (int) sqrt(dim); // sqdim = N 

  /* ----- Real system matrix Hd_real, write it into Bd ---- */
  // if (mpirank_world == 0) printf("Receiving system Hamiltonian...\n");


  MatSetOption(Bd, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  // Read Hsys from file
  long int nelems = sqdim*sqdim;
  std::vector<double> vals (nelems);
  int skiplines=0;
  std::string testheader = "# Hsys ";
  int success = 0;
  if (mpirank_world == 0) {
    success = read_vector(hamiltonian_file.c_str(), vals.data(), nelems, quietmode, skiplines, testheader);
    if (success != 1){
      printf("# ERROR: Did not receive system Hamiltonian.\n");
      exit(1);
    }
  }
  MPI_Bcast(vals.data(), nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Iterate over all elements*/
  for (size_t i = 0; i<vals.size(); i++) {
    if (fabs(vals[i])<1e-14) continue; // Skip zeros

    // Get position in the Bd matrix
    int row = i % sqdim;
    int col = i / sqdim;

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
}

void PythonInterface::receiveHc(int noscillators, std::vector<std::vector<Mat>>& Ac_vec, std::vector<std::vector<Mat>>& Bc_vec){
  PetscInt ilow, iupp;
  int success;
  std::string testheader;

  // if (mpirank_world == 0) printf("Receiving control Hamiltonian terms...\n");

  /* Get the dimensions right */
  int sqdim = dim_rho; //  N!
  int nelems = dim_rho*dim_rho;

  // Skip first Hd lines in the file
  int skiplines = nelems+1; // nelems for Hsys and +1 for the first comment line

  /* Iterate over oscillators */
  for (int k=0; k<noscillators; k++){
    MatSetOption(Ac_vec[k][0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetOption(Bc_vec[k][0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

    /* Read real part from file */
    std::vector<double> vals (nelems);
    testheader = "# Oscillator " + std::to_string(k) + " Hc_real";
    if (mpirank_world == 0) {
      success = read_vector(hamiltonian_file.c_str(), vals.data(), nelems, quietmode, skiplines, testheader);
      if (success==1) skiplines += nelems+1;
    }
    MPI_Bcast(vals.data(), nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // printf("Received ioscid %d, Hc_real = \n", k);
    // for (int m=0; m<vals.size(); m++){
    //   printf("%d %f\n", m, vals[m]);
    // }

    // Iterate over received elements and place into Bc_vec
    MatGetOwnershipRange(Bc_vec[k][0], &ilow, &iupp);
    for (size_t l = 0; l<vals.size(); l++) {
      if (fabs(vals[l])<1e-14) continue; // Skip zeros

      // Get position in the Bc matrix
      int row = l % sqdim;
      int col = l / sqdim;

      if (lindbladtype == LindbladType::NONE){
        // Schroedinger: Assemble - B_c  
        double val = -1.*vals[l];
        if (ilow <= row && row < iupp) MatSetValue(Bc_vec[k][0], row, col, val, ADD_VALUES);
      } else {
        // Lindblad: Assemble -I_N \kron B_c + B_c \kron I_N 
        for (int m=0; m<sqdim; m++){
          // first place all -v_ij in the -I_N\kron B_c term:
          int rowm = row + sqdim * m;
          int colm = col + sqdim * m;
          double val = -1.*vals[l];
          if (ilow <= rowm && rowm < iupp) MatSetValue(Bc_vec[k][0], rowm, colm, val, ADD_VALUES);
          // Then add v_ij in the B_d^T \kron I_N term:
          rowm = col*sqdim + m;   // transpose!
          colm = row*sqdim + m;
          val = vals[l];
          if (ilow <= rowm && rowm < iupp) MatSetValue(Bc_vec[k][0], rowm, colm, val, ADD_VALUES);
        }
      }
    } // end of elements of Hc[k][i] real 

    /* Read imaginary part from file */
    testheader = "# Oscillator " + std::to_string(k) + " Hc_imag";
    if (mpirank_world == 0) {
      success = read_vector(hamiltonian_file.c_str(), vals.data(), nelems, quietmode, skiplines, testheader);
      if (success==1) skiplines += nelems+1;
    }
    MPI_Bcast(vals.data(), nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // printf("Received ioscid %d, Hc_imag = \n", k);
    // for (int m=0; m<vals.size(); m++){
    //   printf("%d %f\n", m, vals[m]);
    // }

    // Iterate over received vals and place into Ac_vec
    MatGetOwnershipRange(Ac_vec[k][0], &ilow, &iupp);
    for (size_t l = 0; l<vals.size(); l++) {
      if (fabs(vals[l])<1e-14) continue; // Skip zeros

      // Get position in the Ac matrix
      int row = l % sqdim;
      int col = l / sqdim;

      if (lindbladtype == LindbladType::NONE){
        // Schroedinger: Assemble A_c  
        double val = vals[l];
        if (ilow <= row && row < iupp) MatSetValue(Ac_vec[k][0], row, col, val, ADD_VALUES);
      } else {
        // Lindblad: Assemble I_N \kron B_c - B_c \kron I_N 
        for (int m=0; m<sqdim; m++){
          // first place all -v_ij in the -I_N\kron B_c term:
          int rowm = row + sqdim * m;
          int colm = col + sqdim * m;
          double val = vals[l];
          if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][0], rowm, colm, val, ADD_VALUES);
          // Then add v_ij in the B_d^T \kron I_N term:
          rowm = col*sqdim + m;   // transpose!
          colm = row*sqdim + m;
          val = -1.0*vals[l];
          if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][0], rowm, colm, val, ADD_VALUES);
        }
      }
    } // end of elements of Hc[k][i] imag
  } // end of k loop for oscillators
}