#include "learning.hpp"


Learning::Learning(const int dim_, LindbladType lindbladtype_){
  dim = dim_;     // Dimension of the state vector (N^2 for Lindblad)
  lindbladtype = lindbladtype_;

  // Get dimension of the Hilbert space (N)
  dim_rho = dim;
  if (lindbladtype != LindbladType::NONE){
    dim_rho = sqrt(dim); 
  }

  /* Set an initial guess for the learnable parameters */
  // learnable Hamiltonian initialization
  int nparams_H = dim-1;  // N^2-1 many
  for (int i=0; i<nparams_H; i++){
    double val = 0.0;                // TODO
    learn_params_H.push_back(val);
  }
  // learnable Collapse operators
  int nparams_C = dim*(dim-1)/2;  // N^2(N^2-1)/2  many
  for (int i=0; i<nparams_C; i++){
    double val = 0.0;                // TODO
    learn_params_C.push_back(val);
  }


  /* Create Gellman matrices */

  /* 1) myG = - I_N \kron sigma_jk^Re + sigma_jk^Re \kron I_N  where  sigma_jk^re = |j><k| + |k><j| */
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
      } else { // Lindbladsolver
        // first part: -I\kron sigma_jk^Re
        for (int i=0; i<dim_rho; i++){
          int row = i*dim_rho + j;
          int col = i*dim_rho + k;
          MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
        }
        // second part: +sigma_jk^Re \kron I
        for (int i=0; i<dim_rho; i++){
          int row = j*dim_rho + i;
          int col = k*dim_rho + i;
          MatSetValue(myG, row, col, 1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, 1.0, INSERT_VALUES);
        }
      }

      MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
      GellmannMats.push_back(myG);
    }
  }

  /* 2) myG = - I_N \kron sigma_jk^Im + sigma_jk^Im \kron I_N  where  sigma_jk^Im = |j><k| - |k><j| */
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
      } else { // Lindblad solver 
        // first part: -I\kron sigma_jk^Im
        for (int i=0; i<dim_rho; i++){
          int row = i*dim_rho + j;
          int col = i*dim_rho + k;
          MatSetValue(myG, row, col, -1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, +1.0, INSERT_VALUES);
        }
        // second part: +sigma_jk^Im \kron I
        for (int i=0; i<dim_rho; i++){
          int row = j*dim_rho + i;
          int col = k*dim_rho + i;
          MatSetValue(myG, row, col, 1.0, INSERT_VALUES);
          MatSetValue(myG, col, row, -1.0, INSERT_VALUES);
        }
      }
      MatAssemblyBegin(myG, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(myG, MAT_FINAL_ASSEMBLY);
      GellmannMats.push_back(myG);
    }
  }

 /* 3) myG = - I_N \kron sigma_l + sigma_l \kron I_N  where  sigma_l = ... */
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
    GellmannMats.push_back(myG);
  }
  printf("Hey dim=%d, dim_rho=%d, Mats %d\n", dim, dim_rho, GellmannMats.size());
  for (int i=0; i<GellmannMats.size(); i++){
    printf("i=%d\n", i);
    MatView(GellmannMats[i], NULL);
  }
  exit(1);
 
}

Learning::~Learning(){
  learn_params_H.clear();
  learn_params_C.clear();

  for (int i=0; i<GellmannMats.size(); i++){
    MatDestroy(&GellmannMats[i]);
  }
  GellmannMats.clear();
}