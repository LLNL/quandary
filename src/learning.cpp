#include "learning.hpp"


Learning::Learning(std::vector<int>& nlevels, LindbladType lindbladtype_, std::vector<std::string>& learninit_str, std::string data_name, double data_dtAWG_, int data_ntime_, std::default_random_engine rand_engine, bool quietmode){
  lindbladtype = lindbladtype_;
  data_dtAWG = data_dtAWG_;
  data_ntime = data_ntime_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);

  // Get dimension of the Hilbert space (dim_rho = N) and dimension of the state variable (dim = N or N^2 for Schroedinger or Lindblad solver)
  dim_rho = 1;
  for (int i=0; i<nlevels.size(); i++){
    dim_rho *= nlevels[i];
  }
  dim = dim_rho;
  if (lindbladtype != LindbladType::NONE){
    dim = dim_rho*dim_rho; 
  }

  /* Load trajectory data from file */
  if (dim_rho > 0) {
    std::ifstream infile_re;
    std::ifstream infile_im;
    infile_re.open(data_name + "_re.dat", std::ifstream::in);
    infile_im.open(data_name + "_im.dat", std::ifstream::in);
    if(infile_re.fail() || infile_im.fail() ) {// checks to see if file opended 
        std::cout << "\n ERROR loading learning data file\n" << std::endl; 
        std::cout << data_name + "_re.dat" << std::endl;
        exit(1);
    } 
    // Iterate over each line
    for (int n = 0; n <data_ntime; n++) {
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, PETSC_DECIDE, 2*dim);
      VecSetFromOptions(state);

      // Iterate over columns
      double val_re, val_im;
      infile_re >> val_re; // first element is time
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
    }
	  infile_re.close();
	  infile_im.close();

    // // TEST what was loaded
    // printf("\nDATA POINTS:\n");
    // for (int i=0; i<data.size(); i++){
    //   VecView(data[i], NULL);
    // }
    // printf("END DATA POINTS.\n\n");
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
  if (nbasis > 0) {
    if (learninit_str[0].compare("file") == 0 ) {
      // Read parameter from file

      assert(learninit_str.size()>1);
      std::vector<double> initguess_fromfile(nbasis, 0.0);
      if (mpirank_world == 0) {
        read_vector(learninit_str[1].c_str(), initguess_fromfile.data(), nbasis, quietmode);
      }
      MPI_Bcast(initguess_fromfile.data(), nbasis, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // Store parameters: First for sigma_jk^RE, they go with first part of Gellmann_B. Then for sigma_jk^IM, they go with first part of Gellmann_A. Then for sigma_l, they go with last part of Gellmann_B.
      int nsigma = dim_rho*(dim_rho-1)/2;
      for (int i=0; i<nsigma; i++){
        learnparamsH_B.push_back(initguess_fromfile[i] * 2.0*M_PI); // radians
      }
      for (int i=nsigma; i<2*nsigma; i++){
        learnparamsH_A.push_back(initguess_fromfile[i] * 2.0*M_PI); // radians
      }
      for (int i=2*nsigma; i<nbasis; i++){
        learnparamsH_B.push_back(initguess_fromfile[i] * 2.0*M_PI); // radians
      }

    } else if (learninit_str[0].compare("random") == 0 ) {
      // Set uniform random parameters in [0,amp)

      assert(learninit_str.size()>1);
      double amp = atof(learninit_str[1].c_str());
      std::uniform_real_distribution<double> unit_dist(0.0, amp);
      for (int i=0; i<GellmannMats_A.size(); i++){
        learnparamsH_A.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
      }
      for (int i=0; i<GellmannMats_B.size(); i++){
        learnparamsH_B.push_back(unit_dist(rand_engine) * 2.0*M_PI); // radians
      }
    } else if (learninit_str[0].compare("constant") == 0 ) {
      // Set constant amp
      assert(learninit_str.size()>1);
      double amp = atof(learninit_str[1].c_str());
      for (int i=0; i<GellmannMats_A.size(); i++){
        learnparamsH_A.push_back(amp * 2.0*M_PI);
      }
      for (int i=0; i<GellmannMats_B.size(); i++){
        learnparamsH_B.push_back(amp * 2.0*M_PI);
      }
    } else {
      printf("ERROR: Wrong configuration for learnable parameter initialization. Choose 'file, <pathtofile>', or 'random, <amplitude>'\n");
      exit(1);
    }

    // for (int i=0; i<learnparamsH_A.size(); i++)
    //   printf("  paramA %d = %f\n", i, learnparamsH_A[i]);
    // for (int i=0; i<learnparamsH_B.size(); i++)
    //   printf("  paramB %d = %f\n", i, learnparamsH_B[i]);
    
    assert(GellmannMats_A.size() == learnparamsH_A.size());
    assert(GellmannMats_B.size() == learnparamsH_B.size());

    // Create auxiliary vector needed for MatMult.
    MatCreateVecs(GellmannMats_A[0], &aux, NULL);
  }
}

Learning::~Learning(){
  learnparamsH_A.clear();
  learnparamsH_B.clear();

  for (int i=0; i<data.size(); i++){
    VecDestroy(&data[i]);
  }
  data.clear();

  for (int i=0; i<GellmannMats_A.size(); i++){
    MatDestroy(&GellmannMats_A[i]);
  }
  for (int i=0; i<GellmannMats_B.size(); i++){
    MatDestroy(&GellmannMats_B[i]);
  }
  GellmannMats_A.clear();
  GellmannMats_B.clear();
  if (nbasis > 0) {
    VecDestroy(&aux);
  }
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
    // uout -= learnparamB * GellmannB * v
    MatMult(GellmannMats_B[i], v, aux);
    VecAXPY(uout, -1.*learnparamsH_B[i], aux); 
    // vout += learnparamB * GellmannB * u
    MatMult(GellmannMats_B[i], u, aux);
    VecAXPY(vout, learnparamsH_B[i], aux);
  }
}


void Learning::applyLearningTerms_diff(Vec u, Vec v, Vec uout, Vec vout){

  // Real parts of (-i * H)
  for (int i=0; i< GellmannMats_A.size(); i++){
    // uout += learnparamA * GellmannA^T * u
    MatMultTranspose(GellmannMats_A[i], u, aux);
    VecAXPY(uout, learnparamsH_A[i], aux); 
    // vout += learnparamA * GellmannA^T * v
    MatMultTranspose(GellmannMats_A[i], v, aux);
    VecAXPY(vout, learnparamsH_A[i], aux);
  }
  // Imaginary parts of (-i * H)
  for (int i=0; i< GellmannMats_B.size(); i++){
    // uout += learnparamB * GellmannB^T * v
    MatMultTranspose(GellmannMats_B[i], v, aux);
    VecAXPY(uout, learnparamsH_B[i], aux); 
    // vout -= learnparamB * GellmannB^T * u
    MatMultTranspose(GellmannMats_B[i], u, aux);
    VecAXPY(vout, -1.*learnparamsH_B[i], aux);
  }
}


void Learning::getLearnOperator(Mat* A, Mat* B){

  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim, dim, NULL, A);
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, dim, dim, NULL, B);
  MatSetUp(*A);
  MatSetUp(*B);
  MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY);

  for (int i=0; i<GellmannMats_A.size(); i++) {
    MatAXPY(*A, learnparamsH_A[i], GellmannMats_A[i], SUBSET_NONZERO_PATTERN);
  }
  for (int i=0; i<GellmannMats_B.size(); i++) {
    MatAXPY(*B, learnparamsH_B[i], GellmannMats_B[i], SUBSET_NONZERO_PATTERN);
  }
}


void Learning::setLearnParams(const Vec x){

  const PetscScalar* ptr;
  VecGetArrayRead(x, &ptr);

  /* Storage of parameters in x:
   * First all sigma_jk^RE (in GellmannB), 
   * then all sigma_ij^IM (in GellmannA), 
   * then all sigma_l (in GellmannB) */

  int nsigma = dim_rho*(dim_rho-1)/2;
  for (int i=0; i<nsigma; i++) {
    learnparamsH_B[i] = ptr[i];
    learnparamsH_A[i] = ptr[i+nsigma];
  }
  for (int i=0; i<dim_rho-1; i++){
    learnparamsH_B[i+nsigma] = ptr[i+2*nsigma];
  }

  VecRestoreArrayRead(x, &ptr);

  // for (int i=0; i<learnparamsH_A.size(); i++){
  //   printf("A %1.14e% \n", learnparamsH_A[i]);
  // }
  // for (int i=0; i<learnparamsH_B.size(); i++){
  //   printf("B %1.14e \n", learnparamsH_B[i]);
  // }
}

void Learning::getLearnParams(double* x){
  int nsigma = dim_rho*(dim_rho-1)/2;
  for (int i=0; i<nsigma; i++) {
    x[i]        = learnparamsH_B[i];
    x[i+nsigma] = learnparamsH_A[i];
  }
  for (int i=0; i<dim_rho-1; i++){
    x[i+2*nsigma] = learnparamsH_B[i+nsigma];
  }
}

void Learning::dRHSdp(Vec grad, Vec u, Vec v, double alpha, Vec ubar, Vec vbar){
  double uAubar, vAvbar, vBubar, uBvbar;

  // gamma_bar_A += alpha * (  u^t sigma_A^t ubar + v^t sigma_A^t vbar )
  // gamma_bar_B += alpha * ( -v^t sigma_B^t ubar + u^t sigma_B^t vbar )

  int nsigma = dim_rho*(dim_rho-1)/2;
  for (int i=0; i< nsigma; i++){

    MatMult(GellmannMats_B[i], u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(GellmannMats_B[i], v, aux); VecDot(aux, ubar, &vBubar);
    VecSetValue(grad, i, alpha*(-vBubar + uBvbar), ADD_VALUES);

    MatMult(GellmannMats_A[i], u, aux); VecDot(aux, ubar, &uAubar);
    MatMult(GellmannMats_A[i], v, aux); VecDot(aux, vbar, &vAvbar);
    VecSetValue(grad, i+nsigma, alpha*(uAubar + vAvbar), ADD_VALUES);
  }

  for (int i=0; i<dim_rho-1; i++){
    MatMult(GellmannMats_B[i+nsigma], u, aux); VecDot(aux, vbar, &uBvbar);
    MatMult(GellmannMats_B[i+nsigma], v, aux); VecDot(aux, ubar, &vBubar);
    VecSetValue(grad, i+2*nsigma, alpha*(-vBubar + uBvbar), ADD_VALUES);
  }

  VecAssemblyBegin(grad);
  VecAssemblyEnd(grad);
}