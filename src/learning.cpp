#include "learning.hpp"


Learning::Learning(const int dim_, LindbladType lindbladtype_){
  dim = dim_;     // Dimension of the state vector (N^2 for Lindblad)
  lindbladtype = lindbladtype_;

  // Get dimension of the Hilbert space (N)
  dim_rho = dim;
  if (lindbladtype != LindbladType::NONE){
    dim_rho = sqrt(dim); 
  }

  /* Create generalized Gellman matrices, multiplied by (-i) and shifted s.t. G_00=0 */

  /* 1) Real offdiagonal Gellman matrices:  sigma_jk^re = |j><k| + |k><j| 
        Note: (-i)sigma_jk^RE is purely imaginary, hence into Gellman_B = Im(H) */
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      Mat myG;
      MatCreate(PETSC_COMM_WORLD, &myG);
      MatSetType(myG, MATMPIAIJ);
      MatSetSizes(myG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      // MatMPIAIJSetPreallocation(myG, 2, NULL, 2, NULL);  // How many to allocate? Split into diag and off diag per proc.
      MatSetUp(myG);

      if (lindbladtype==LindbladType::NONE) { // Schroedinger solver
        int row = j;
        int col = k;
        MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
        MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
      } else {
        // For Lindblad: I_N \kron (-i)sigma_jk^Re - (-i)sigma_jk^Re \kron I_N  */
        //  -I\kron sigma_jk^Re
        for (int i=0; i<dim_rho; i++){
          int row = i*dim_rho + j;
          int col = i*dim_rho + k;
          MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
        }
        // +sigma_jk^Re \kron I
        for (int i=0; i<dim_rho; i++){
          int row = j*dim_rho + i;
          int col = k*dim_rho + i;
          MatSetValue(myG, row, col, 1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, 1.0, INSERT_VALUES);
        }
      }

      MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
      GellmannMats_B.push_back(myG);
    }
  }

  /* 2) myG = - I_N \kron sigma_jk^Im + sigma_jk^Im \kron I_N  where  sigma_jk^Im = |j><k| - |k><j| */
  /* 2) Imaginary offdiagonal Gellman matrices:  sigma_jk^im = -i|j><k| + i|k><j| 
        Note: (-i)sigma_jk^IM is real, hence into Gellman_A = Re(H) */
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      Mat myG;
      MatCreate(PETSC_COMM_WORLD, &myG);
      MatSetType(myG, MATMPIAIJ);
      MatSetSizes(myG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
      // MatMPIAIJSetPreallocation(myG, 2, NULL, 2, NULL);  // How many to allocate? Split into diag and off diag per proc.
      MatSetUp(myG);

      if (lindbladtype==LindbladType::NONE) { // Schroedinger solver
        int row = j;
        int col = k;
        MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
        MatSetValue(myG, col, row, +1.0, INSERT_VALUES);
      } else {
        // For Lindblad: I_N \kron (-i)sigma_jk^Im - (-i)sigma_jk^Ie \kron I_N  */
        //  -I\kron sigma_jk^Im
        for (int i=0; i<dim_rho; i++){
          int row = i*dim_rho + j;
          int col = i*dim_rho + k;
          MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, +1.0, INSERT_VALUES);
        }
        // +sigma_jk^Im \kron I
        for (int i=0; i<dim_rho; i++){
          int row = j*dim_rho + i;
          int col = k*dim_rho + i;
          MatSetValue(myG, row, col, 1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
        }
      }
      MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
      GellmannMats_A.push_back(myG);
    }
  }

  /* 3) Real diagonal Gellman matrices:  sigma_l^Re = (2/(l(l+1))(sum_j|j><j| - l|l><l|) 
        Note: (-i)sigma_l^RE is purely imaginary, hence into Gellman_B = Im(H) */
  for (int l=1; l<dim_rho; l++){
    Mat myG;
    MatCreate(PETSC_COMM_WORLD, &myG);
    MatSetType(myG, MATMPIAIJ);
    MatSetSizes(myG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    // MatMPIAIJSetPreallocation(myG, 2, NULL, 2, NULL);  // How many to allocate? Split into diag and off diag per proc.
    MatSetUp(myG);

    double prefactor = sqrt(2.0/(l*(l+1)));

    if (lindbladtype==LindbladType::NONE) { // Schroedinger solver
      int row = l;
      MatSetValue(myG, row, row, +1.0*l*prefactor, ADD_VALUES);
      for (int j=l; j<dim_rho; j++){
        int row = j;
        MatSetValue(myG, row, row, +1.0*prefactor, ADD_VALUES);
      }
    } else { // Lindblad solver
      // first part: -I\kron sigma_l
      for (int i=0; i<dim_rho; i++){
        int row = i*dim_rho + l;
        MatSetValue(myG, row, row, +1.0*l*prefactor, ADD_VALUES);
        for (int j=l; j<dim_rho; j++){
          int row = i*dim_rho + j;
          MatSetValue(myG, row, row, +1.0*prefactor, ADD_VALUES);
        }
      }
      // second part: +sigma_l \kron I
      for (int i=0; i<dim_rho; i++){
        int row = l*dim_rho + i;
        MatSetValue(myG, row, row, -1.0*l*prefactor, ADD_VALUES);
        for (int j=l; j<dim_rho; j++){
          int row = j*dim_rho + i;
          MatSetValue(myG, row, row, -1.0*prefactor, ADD_VALUES);
        }
      }
    }

    MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
    GellmannMats_B.push_back(myG);
  }
  // printf("Learnable basis matrices for dim=%d, dim_rho=%d, A-Mats:%d, B-Mats: %d\n", dim, dim_rho, GellmannMats_A.size(), GellmannMats_B.size() );
  // for (int i=0; i<GellmannMats_A.size(); i++){
  //   printf("Gellman A: i=%d\n", i);
  //   MatView(GellmannMats_A[i], NULL);
  // }
  // for (int i=0; i<GellmannMats_B.size(); i++){
  //   printf("Gellman B: i=%d\n", i);
  //   MatView(GellmannMats_B[i], NULL);
  // }
  // exit(1);

  // Store the number of basis elements 
  nbasis = GellmannMats_A.size() + GellmannMats_B.size();

  /* Set an initial guess for the learnable Hamiltonian parameters */
  for (int i=0; i<GellmannMats_A.size(); i++){
    double val = 0.0;                // TODO
    learnparamsH_A.push_back(val);
  }
  for (int i=0; i<GellmannMats_B.size(); i++){
    double val = 0.0;                // TODO
    learnparamsH_B.push_back(val);
  }

  // Create auxiliary vector 
  if (GellmannMats_A.size()>0) {
    MatCreateVecs(GellmannMats_A[0], &aux, NULL);
  }
}

Learning::~Learning(){
  learnparamsH_A.clear();
  learnparamsH_B.clear();

  for (int i=0; i<GellmannMats_A.size(); i++){
    MatDestroy(&GellmannMats_A[i]);
  }
  for (int i=0; i<GellmannMats_B.size(); i++){
    MatDestroy(&GellmannMats_B[i]);
  }
  GellmannMats_A.clear();
  GellmannMats_B.clear();
}

void Learning::applyLearningTerms(Vec u, Vec v, Vec uout, Vec vout){
      // Real parts of (-i * H)
    for (int i=0; i< GellmannMats_A.size(); i++){
      // uout += learnparamA * GellmannA * u
      MatMult(GellmannMats_A[i], u, aux);
      VecAXPY(uout, learnparamsH_A[i], aux); 
      // vout += learnparamA * GellmannA * v
      MatMult(GellmannMats_A[i], v, aux);
      VecAXPY(vout, learnparamsH_A[i], aux);
    }
    // Imaginary parts of (-i * H)
    for (int i=0; i< GellmannMats_B.size(); i++){
      // uout -= learnparamB * GellmannB * u
      MatMult(GellmannMats_B[i], v, aux);
      VecAXPY(uout, -1.*learnparamsH_B[i], aux); 
      // vout += learnparamB * GellmannB * u
      MatMult(GellmannMats_B[i], u, aux);
      VecAXPY(vout, learnparamsH_B[i], aux);
    }
}