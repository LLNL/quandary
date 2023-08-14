#include "pythoninterface.hpp"

PythonInterface::PythonInterface(){
}


PythonInterface::PythonInterface(std::string hamiltonian_file_, LindbladType lindbladtype_, int dim_rho_) {

  lindbladtype = lindbladtype_;
  dim_rho = dim_rho_;
  hamiltonian_file = hamiltonian_file_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
}

PythonInterface::~PythonInterface(){
}

void PythonInterface::receiveHsys(Mat& Bd){

  /* Get matrix sizes */ 
  PetscInt dim = 0;
  MatGetSize(Bd, &dim, NULL); // could be N^2 or N
  
  /* Get size of Hilbert space */
  int sqdim = dim;
  if (lindbladtype != LindbladType::NONE) sqdim = (int) sqrt(dim); // sqdim = N 

  /* ----- Real system matrix Hd_real, write it into Bd ---- */
  if (mpirank_world == 0) printf("Receiving system Hamiltonian...\n");

  // Bd had been allocated before. Destroy and reallocate it here. 
  MatDestroy(&Bd);
  MatCreate(PETSC_COMM_WORLD, &Bd);
  MatSetType(Bd, MATMPIAIJ);
  MatSetSizes(Bd, PETSC_DECIDE, PETSC_DECIDE, dim, dim); // dim = N^2 for Lindblad, N for Schroedinger
  MatSetUp(Bd);
  MatSetFromOptions(Bd);
  PetscInt ilow, iupp;
  MatGetOwnershipRange(Bd, &ilow, &iupp);

  // read Hsys from file
  long int nelems = sqdim*sqdim;
  std::vector<double> vals (nelems);
  int skiplines=1;
  if (mpirank_world == 0) read_vector(hamiltonian_file.c_str(), vals.data(), nelems, false, skiplines);
  MPI_Bcast(vals.data(), nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Place vars into Bd */
  for (int i = 0; i<vals.size(); i++) {

    // Skip all non-zero elements to preserve sparse storage. 
    if (fabs(vals[i])<1e-14) continue;

    // Get position in the Bd matrix
    int row = i % sqdim;
    int col = i / sqdim;
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
}

void PythonInterface::receiveHc(int noscillators, std::vector<std::vector<Mat>>& Ac_vec, std::vector<std::vector<Mat>>& Bc_vec){
  PetscInt ilow, iupp;

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
  int nelems = dim_rho*dim_rho;

  // Skip first Hd lines in the file
  int skiplines = nelems+1; // +1 for the first comment line

  /* Iterate over oscillators */
  for (int k=0; k<noscillators; k++){

    // Check number of control terms for this oscillator
    int ncontrol = 1; // Assume for one control term per oscillator (real and imag)
    ncontrol_real[k] = ncontrol;
    ncontrol_imag[k] = ncontrol;

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
      MatGetOwnershipRange(Bc_vec[k][i], &ilow, &iupp);

      /* Read real part from file */
      std::vector<double> vals (nelems);
      if (mpirank_world == 0) read_vector(hamiltonian_file.c_str(), vals.data(), nelems, false, skiplines);
      MPI_Bcast(vals.data(), nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      /* Assemble -I_N \kron Hc^k_i + Hc^k_i \kron I_N (Lindblad) or -Hc^k_i (Schroedinger) */

      // Set vals in real part of Hc for this oscillator
      for (int l = 0; l<vals.size(); l++) {

        // Skip all non-zero elements to preserve sparse storage. 
        if (fabs(vals[l])<1e-14) continue;

        // Get position in the Bc matrix
        int row = l % sqdim;
        int col = l / sqdim;

        if (lindbladtype == LindbladType::NONE){
          // Schroedinger
          // Assemble - B_c  
          double val = -1.*vals[l];
          if (ilow <= row && row < iupp) MatSetValue(Bc_vec[k][i], row, col, val, ADD_VALUES);
        } else {
          // Lindblad
          // Assemble -I_N \kron B_c + B_c \kron I_N 
          for (int m=0; m<sqdim; m++){
            // first place all -v_ij in the -I_N\kron B_c term:
            int rowm = row + sqdim * m;
            int colm = col + sqdim * m;
            double val = -1.*vals[l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Bc_vec[k][i], rowm, colm, val, ADD_VALUES);
            // Then add v_ij in the B_d^T \kron I_N term:
            rowm = col*sqdim + m;   // transpose!
            colm = row*sqdim + m;
            val = vals[l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Bc_vec[k][i], rowm, colm, val, ADD_VALUES);
          }
        }
      } // end of elements of Hc[k][i] real 
      MatAssemblyBegin(Bc_vec[k][i], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Bc_vec[k][i], MAT_FINAL_ASSEMBLY);

      /* IMAGINARY PART */

      // Create a new control matrix
      Mat myAcMat_kl;
      Ac_vec[k].push_back(myAcMat_kl);

      MatCreate(PETSC_COMM_WORLD, &(Ac_vec[k][i]));
      MatSetType(Ac_vec[k][i], MATMPIAIJ);
      MatSetSizes(Ac_vec[k][i], PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      MatSetUp(Ac_vec[k][i]);
      MatSetFromOptions(Ac_vec[k][i]);
      MatGetOwnershipRange(Ac_vec[k][i], &ilow, &iupp);

      /* Read imaginary part from file */
      skiplines+=nelems+1; // Skip system Hamiltonian and real Hc
      if (mpirank_world == 0) read_vector(hamiltonian_file.c_str(), vals.data(), nelems, false, skiplines);
      MPI_Bcast(vals.data(), nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Assemble I_N \kron Hc^k_i - Hc^k_i \kron I_N (Lindblad) or Hc^k_i (Schroedinger)

      // Set vals in imag part of Hc for this oscillator
      for (int l = 0; l<vals.size(); l++) {

        // Skip all non-zero elements to preserve sparse storage. 
        if (fabs(vals[l])<1e-14) continue;

        // Get position in the Bc matrix
        int row = l % sqdim;
        int col = l / sqdim;

        if (lindbladtype == LindbladType::NONE){
          // Schroedinger
          // Assemble A_c  
          double val = vals[l];
          if (ilow <= row && row < iupp) MatSetValue(Ac_vec[k][i], row, col, val, ADD_VALUES);
        } else {
          // Lindblad
          // Assemble I_N \kron B_c - B_c \kron I_N 
          for (int m=0; m<sqdim; m++){
            // first place all -v_ij in the -I_N\kron B_c term:
            int rowm = row + sqdim * m;
            int colm = col + sqdim * m;
            double val = vals[l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][i], rowm, colm, val, ADD_VALUES);
            // Then add v_ij in the B_d^T \kron I_N term:
            rowm = col*sqdim + m;   // transpose!
            colm = row*sqdim + m;
            val = -1.0*vals[l];
            if (ilow <= rowm && rowm < iupp) MatSetValue(Ac_vec[k][i], rowm, colm, val, ADD_VALUES);
          }
        }
      } // end of elements of Hc[k][i] imag
      MatAssemblyBegin(Ac_vec[k][i], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(Ac_vec[k][i], MAT_FINAL_ASSEMBLY);

    } // end of i loop for control terms
  } // end of k loop for oscillators
}