#include "util.hpp"

void createGellmannMats(int dim_rho, bool upper_only, bool real_only, bool shifted_diag, bool includeIdentity, std::vector<Mat>& Mats_Re, std::vector<Mat>& Mats_Im){

  /* First empty out the vectors, if needed */
  for (int i=0; i<Mats_Re.size(); i++) MatDestroy(&Mats_Re[i]);
  for (int i=0; i<Mats_Im.size(); i++) MatDestroy(&Mats_Im[i]);

  /* Put the identity first, if needed */
  if (includeIdentity){
    Mat G_re;
    MatCreate(PETSC_COMM_WORLD, &G_re);
    MatSetType(G_re, MATSEQAIJ);
    MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(G_re);
    for (int i=0; i<dim_rho; i++){
      MatSetValue(G_re, i, i, 1.0, INSERT_VALUES);
    }
    MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
    Mats_Re.push_back(G_re);
  }

  /* Create all offdiagonal matrices (re and im)*/
  for (int j=0; j<dim_rho; j++){
    for (int k=j+1; k<dim_rho; k++){
      /* Real sigma_jk^re = |j><k| + |k><j| */ 
      Mat G_re;
      MatCreate(PETSC_COMM_WORLD, &G_re);
      MatSetType(G_re, MATSEQAIJ);
      MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
      MatSetUp(G_re);
      MatSetValue(G_re, j, k, 1.0, INSERT_VALUES);
      if (!upper_only) MatSetValue(G_re, k, j, 1.0, INSERT_VALUES);
      MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
      Mats_Re.push_back(G_re);

      /* Imaginary sigma_jk^im = -i|j><k| + i|k><j| */ 
      if (!real_only) {
        Mat G_im;
        MatCreate(PETSC_COMM_WORLD, &G_im);
        MatSetType(G_im, MATSEQAIJ);
        MatSetSizes(G_im, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
        MatSetUp(G_im);
        MatSetValue(G_im, j, k, -1.0, INSERT_VALUES);
        if (!upper_only) MatSetValue(G_im, k, j, +1.0, INSERT_VALUES);
        /* Assemble and store */
        MatAssemblyBegin(G_im, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(G_im, MAT_FINAL_ASSEMBLY);
        Mats_Im.push_back(G_im);
      }
    }
  }

  /* All diagonal matrices  */
  for (int l=1; l<dim_rho; l++){
    Mat G_re;
    MatCreate(PETSC_COMM_WORLD, &G_re);
    MatSetType(G_re, MATSEQAIJ);
    MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(G_re);

    /* Diagonal mats 
     *  shifted:     sigma_l^Re = (2/(l(l+1))( sum_{j=l,...,N-1} -|j><j| - l|l><l|)
     *  not shifted: sigma_l^Re = (2/(l(l+1))( sum_{j=0,...,l-1}  |j><j| - l|l><l|) 
     */
    double factor = sqrt(2.0/(l*(l+1)));
    MatSetValue(G_re, l, l, -1.0*l*factor, ADD_VALUES);
    if (shifted_diag) {      
      for (int j=l; j<dim_rho; j++){
        MatSetValue(G_re, j, j, -1.0*factor, ADD_VALUES);
      }
    } else {  
      for (int j=0; j<l; j++){
        MatSetValue(G_re, j, j, factor, ADD_VALUES);
      }
    }
    MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
    Mats_Re.push_back(G_re);
  }
}


void createEijBasisMats(int dim_rho, bool includeIdentity, std::vector<Mat>& Mats_Re, std::vector<Mat>& Mats_Im){

  /* First empty out the vectors, if needed */
  for (int i=0; i<Mats_Re.size(); i++) MatDestroy(&Mats_Re[i]);
  for (int i=0; i<Mats_Im.size(); i++) MatDestroy(&Mats_Im[i]);

  /* Put the identity first, if needed */
  if (includeIdentity){
    Mat G_re;
    MatCreate(PETSC_COMM_WORLD, &G_re);
    MatSetType(G_re, MATSEQAIJ);
    MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(G_re);
    for (int i=0; i<dim_rho; i++){
      MatSetValue(G_re, i, i, 1.0, INSERT_VALUES);
    }
    MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
    Mats_Re.push_back(G_re);
  }

  /* Create E_ij matrices*/
  for (int i=0; i<dim_rho; i++){
    for (int j=0; j<dim_rho; j++){

      // if (upper_only && j<i) continue;
      // if (shifted_diag && i==0 && j==0) continue; 

      Mat G_re;
      MatCreate(PETSC_COMM_WORLD, &G_re);
      MatSetType(G_re, MATSEQAIJ);
      MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
      MatSetUp(G_re);

      if (i==j){
        // if (i<dim_rho-1) {
        //   double fact = 1.0/sqrt((i+1)*(i+2));
        //   for (int l=0; l<i+1; l++){
        //     MatSetValue(G_re, l, l, 1.0*fact, INSERT_VALUES);
        //   }
        //   MatSetValue(G_re, i+1, i+1, -1.0*(i+1)*fact, INSERT_VALUES);
        //   MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
        //   MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
        //   Mats_Re.push_back(G_re);
        if (i>0) {
          MatSetValue(G_re, i, i, 1.0, INSERT_VALUES);
          MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
          MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
          Mats_Re.push_back(G_re);
        } else { 
          continue; 
        }
      } else {
        MatSetValue(G_re, i, j, 1.0, INSERT_VALUES);
        MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
        Mats_Re.push_back(G_re);
      }
    }
  }
  // for (int i=0; i<Mats_Re.size(); i++){
  //   MatView(Mats_Re[i], NULL);
  // }
  // exit(1);
}


void createDecayBasis_2qubit(int dim_rho, std::vector<Mat>& BasisMats_Re, bool includeIdentity){

  /* Put the identity first, if needed */
  if (includeIdentity){
    Mat G_re;
    MatCreate(PETSC_COMM_WORLD, &G_re);
    MatSetType(G_re, MATSEQAIJ);
    MatSetSizes(G_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(G_re);
    for (int i=0; i<dim_rho; i++){
      MatSetValue(G_re, i, i, 1.0, INSERT_VALUES);
    }
    MatAssemblyBegin(G_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(G_re, MAT_FINAL_ASSEMBLY);
    BasisMats_Re.push_back(G_re);
  }


  /* First decay and decoherence for each qubit */
  Mat G01, G02, G11, G12;
  MatCreate(PETSC_COMM_WORLD, &G01);
  MatCreate(PETSC_COMM_WORLD, &G02);
  MatCreate(PETSC_COMM_WORLD, &G11);
  MatCreate(PETSC_COMM_WORLD, &G12);
  MatSetType(G01, MATSEQAIJ);
  MatSetType(G02, MATSEQAIJ);
  MatSetType(G11, MATSEQAIJ);
  MatSetType(G12, MATSEQAIJ);
  MatSetSizes(G01, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
  MatSetSizes(G02, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
  MatSetSizes(G11, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
  MatSetSizes(G12, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
  MatSetUp(G01);
  MatSetUp(G02);
  MatSetUp(G11);
  MatSetUp(G12);

  MatSetValue(G01, 0, 2, 1.0, ADD_VALUES);
  MatSetValue(G01, 1, 3, 1.0, ADD_VALUES);
  MatSetValue(G02, 2, 2, 1.0, ADD_VALUES);
  MatSetValue(G02, 3, 3, 1.0, ADD_VALUES);
  MatSetValue(G11, 0, 1, 1.0, ADD_VALUES);
  MatSetValue(G11, 2, 3, 1.0, ADD_VALUES);
  MatSetValue(G12, 1, 1, 1.0, ADD_VALUES);
  MatSetValue(G12, 3, 3, 1.0, ADD_VALUES);
 
  MatAssemblyBegin(G01, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(G02, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(G11, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(G12, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G01, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G02, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G11, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G12, MAT_FINAL_ASSEMBLY);
  BasisMats_Re.push_back(G01);
  BasisMats_Re.push_back(G02);
  BasisMats_Re.push_back(G11);
  BasisMats_Re.push_back(G12);

  for (int i=1; i<4; i++) {
    Mat Gu, Gl;
    MatCreate(PETSC_COMM_WORLD, &Gu);
    MatCreate(PETSC_COMM_WORLD, &Gl);
    MatSetType(Gu, MATSEQAIJ);
    MatSetType(Gl, MATSEQAIJ);
    MatSetSizes(Gu, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetSizes(Gl, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(Gu);
    MatSetUp(Gl);
    MatSetValue(Gu, 0, i, 1.0, ADD_VALUES);
    MatSetValue(Gl, i, 0, 1.0, ADD_VALUES);
    MatAssemblyBegin(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Gl, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gl, MAT_FINAL_ASSEMBLY);
    BasisMats_Re.push_back(Gu);
    BasisMats_Re.push_back(Gl);
  }

  {
    Mat Gu, Gl;
    MatCreate(PETSC_COMM_WORLD, &Gu);
    MatCreate(PETSC_COMM_WORLD, &Gl);
    MatSetType(Gu, MATSEQAIJ);
    MatSetType(Gl, MATSEQAIJ);
    MatSetSizes(Gu, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetSizes(Gl, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(Gu);
    MatSetUp(Gl);
    MatSetValue(Gu, 1, 2, 1.0, ADD_VALUES);
    MatSetValue(Gl, 2, 1, 1.0, ADD_VALUES);
    MatAssemblyBegin(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Gl, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gl, MAT_FINAL_ASSEMBLY);
    BasisMats_Re.push_back(Gu);
    BasisMats_Re.push_back(Gl);
  }
  {
    Mat Gu, Gl;
    MatCreate(PETSC_COMM_WORLD, &Gu);
    MatCreate(PETSC_COMM_WORLD, &Gl);
    MatSetType(Gu, MATSEQAIJ);
    MatSetType(Gl, MATSEQAIJ);
    MatSetSizes(Gu, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetSizes(Gl, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(Gu);
    MatSetUp(Gl);
    MatSetValue(Gu, 3, 1, 1.0, ADD_VALUES);
    MatSetValue(Gl, 3, 2, 1.0, ADD_VALUES);
    MatAssemblyBegin(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Gl, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gl, MAT_FINAL_ASSEMBLY);
    BasisMats_Re.push_back(Gu);
    BasisMats_Re.push_back(Gl);
  }
  {
    Mat Gu;
    MatCreate(PETSC_COMM_WORLD, &Gu);
    MatSetType(Gu, MATSEQAIJ);
    MatSetSizes(Gu, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
    MatSetUp(Gu);
    MatSetValue(Gu, 1, 1, 1.0, ADD_VALUES);
    MatAssemblyBegin(Gu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Gu, MAT_FINAL_ASSEMBLY);
    BasisMats_Re.push_back(Gu);
  }
}

double expectedEnergy(const Vec x, LindbladType lindbladtype, std::vector<int> nlevels, int subsystem){
 
  // Compute Hilbertspace dimension and the dimension of the systems following this subsystem
  PetscInt dim=1;
  PetscInt post_dim = 1;
  for (int i=0; i<nlevels.size(); i++){
    dim *= nlevels[i];
    if (i > subsystem) {
      post_dim *= nlevels[i];
    }
  }
  // Sanity check on full Hilbert space dimension:
  PetscInt dim_test;
  VecGetSize(x, &dim_test);
  dim_test = dim_test / 2;  // since x stores real and imaginary numbers separately
  if (lindbladtype != LindbladType::NONE){ // take the square root if lindblad solver -> N
    dim_test = int(sqrt(dim_test)); 
  }
  assert(dim_test == dim);

  /* Get locally owned portion of x */
  PetscInt ilow, iupp, idx_diag_re, idx_diag_im;
  VecGetOwnershipRange(x, &ilow, &iupp);
  double xdiag;

  /* Iterate over diagonal elements to add up expected energy level */
  double expected = 0.0;
  for (int i=0; i<dim; i++) {
    /* Get diagonal element in number operator */
    int num_diag = i;  // for full composite system
    if (subsystem >= 0) { // for a subsystem
      num_diag = i % (nlevels[subsystem]*post_dim);
      num_diag = num_diag / post_dim;
    }

    /* Get diagonal element in rho (real) and sum up */
    if (lindbladtype != LindbladType::NONE){ // Lindblad solver: += i * rho_ii
      idx_diag_re = getIndexReal(getVecID(i,i,dim));
      xdiag = 0.0;
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecGetValues(x, 1, &idx_diag_re, &xdiag);
      expected += num_diag * xdiag;

      // Make sure the diagonal is real. 
      idx_diag_im = getIndexImag(getVecID(i,i,dim));
      if (ilow <= idx_diag_im && idx_diag_im < iupp) VecGetValues(x, 1, &idx_diag_im, &xdiag);
      if (xdiag > 1e-10) {
        printf("WARNING: imaginary number on the diagonal of the density matrix! %1.14e\n", xdiag);
      }
    }
    else { // Schoedinger solver: += i * | psi_i |^2
      idx_diag_re = getIndexReal(i);
      xdiag = 0.0;
      if (ilow <= idx_diag_re && idx_diag_re < iupp) VecGetValues(x, 1, &idx_diag_re, &xdiag);
      expected += num_diag * xdiag * xdiag;
      idx_diag_im = getIndexImag(i);
      xdiag = 0.0;
      if (ilow <= idx_diag_im && idx_diag_im < iupp) VecGetValues(x, 1, &idx_diag_im, &xdiag);
      expected += num_diag * xdiag * xdiag;
    }
  }
  
  /* Sum up from all Petsc processors */
  double myexp = expected;
  MPI_Allreduce(&myexp, &expected, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  return expected;
}

// void expectedEnergy_diff(const Vec x, Vec x_bar, const double obj_bar) {
//   PetscInt dim;
//   VecGetSize(x, &dim);
//   int dimmat;
//   if (lindbladtype != LindbladType::NONE) dimmat = (int) sqrt(dim/2);
//   else dimmat = (int) dim/2;
//   double num_diag, xdiag, val;

//   /* Get locally owned portion of x */
//   PetscInt ilow, iupp, idx_diag_re, idx_diag_im;
//   VecGetOwnershipRange(x, &ilow, &iupp);

//   /* Derivative of projective measure */
//   for (int i=0; i<dimmat; i++) {
//     int num_diag = i % (nlevels*dim_postOsc);
//     num_diag = num_diag / dim_postOsc;
//     if (lindbladtype != LindbladType::NONE) { // Lindblas solver
//       val = num_diag * obj_bar;
//       idx_diag_re = getIndexReal(getVecID(i, i, dimmat));
//       if (ilow <= idx_diag_re && idx_diag_re < iupp) VecSetValues(x_bar, 1, &idx_diag_re, &val, ADD_VALUES);
//     }
//     else {
//       // Real part
//       idx_diag_re = getIndexReal(i);
//       xdiag = 0.0;
//       if (ilow <= idx_diag_re && idx_diag_re < iupp) VecGetValues(x, 1, &idx_diag_re, &xdiag);
//       val = num_diag * xdiag * obj_bar;
//       if (ilow <= idx_diag_re && idx_diag_re < iupp) VecSetValues(x_bar, 1, &idx_diag_re, &val, ADD_VALUES);
//       // Imaginary part
//       idx_diag_im = getIndexImag(i);
//       xdiag = 0.0;
//       if (ilow <= idx_diag_im && idx_diag_im < iupp) VecGetValues(x, 1, &idx_diag_im, &xdiag);
//       val = - num_diag * xdiag * obj_bar; // TODO: Is this a minus or a plus?? 
//       if (ilow <= idx_diag_im && idx_diag_im < iupp) VecSetValues(x_bar, 1, &idx_diag_im, &val, ADD_VALUES);
//     }
//   }
//   VecAssemblyBegin(x_bar); VecAssemblyEnd(x_bar);
// }

void population(const Vec x, LindbladType lindbladtype, std::vector<double> &pop){

  // Compute Hilbertspace dimension -> N
  PetscInt dim;
  VecGetSize(x, &dim);
  int dim_rho = dim / 2;  // since x stores real and imaginary numbers separately
  if (lindbladtype != LindbladType::NONE){ // take the square root if lindblad solver -> N
    dim_rho= int(sqrt(dim_rho)); 
  }

  // Zero out the population vector
  pop.clear();
  pop.resize(dim_rho);

  /* Get locally owned portion of x */
  PetscInt ilow, iupp;
  VecGetOwnershipRange(x, &ilow, &iupp);

  /* Iterate over diagonal elements of the density matrix */
  std::vector<double> mypop(dim_rho, 0.0);
  for (int idiag=0; idiag < dim_rho; idiag++) {
    double popi = 0.0;
    /* Get the diagonal element */
    if (lindbladtype != LindbladType::NONE) { // Lindblad solver
      PetscInt diagID = getIndexReal(getVecID(idiag, idiag, dim_rho));  // Position in vectorized rho
      double val = 0.0;
      if (ilow <= diagID && diagID < iupp)  VecGetValues(x, 1, &diagID, &val);
      popi = val;
    } else {
      PetscInt diagID_re = getIndexReal(idiag);
      PetscInt diagID_im = getIndexImag(idiag);
      double val = 0.0;
      if (ilow <= diagID_re && diagID_re < iupp)  VecGetValues(x, 1, &diagID_re, &val);
      popi = val * val;
      val = 0.0;
      if (ilow <= diagID_im && diagID_im < iupp)  VecGetValues(x, 1, &diagID_im, &val);
      popi += val * val;
    }
    mypop[idiag] = popi;
  } 

  /* Gather poppulation from all Petsc processors and store in the output vector */
  for (int i=0; i<mypop.size(); i++) {pop[i] = mypop[i];}
  MPI_Allreduce(mypop.data(), pop.data(), dim_rho, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
}

double sigmoid(double width, double x){
  return 1.0 / ( 1.0 + exp(-width*x) );
}

double sigmoid_diff(double width, double x){
  return sigmoid(width, x) * (1.0 - sigmoid(width, x)) * width;
}

double getRampFactor(const double time, const double tstart, const double tstop, const double tramp){

    double eps = 1e-4; // Cutoff for sigmoid ramp 
    double steep = log(1./eps - 1.) * 2. / tramp; // steepness of sigmoid such that ramp(x) small for x < - tramp/2
    // printf("steep eval %f\n", steep);

    double rampfactor = 0.0;
    if (time <= tstart + tramp) { // ramp up
      double center = tstart + tramp/2.0;
      // rampfactor = sigmoid(steep, time - center);
      rampfactor =  1.0/tramp * time - tstart/ tramp;
    }
    else if (tstart + tramp <= time && 
            time <= tstop - tramp) { // center
      rampfactor = 1.0;
    }
    else if (time >= tstop - tramp && time <= tstop) { // down
      double center = tstop - tramp/2.0;
      // rampfactor = sigmoid(steep, -(time - center));
      // steep = 1842.048073;
      // steep = 1000.0;
      rampfactor =  -1.0/tramp * time + tstop / tramp;
    }

    // If ramp time is larger than total amount of time, turn off control:
    if (tstop < tstart + 2*tramp) rampfactor = 0.0;

    return rampfactor;
}

double getRampFactor_diff(const double time, const double tstart, const double tstop, const double tramp){

    double eps = 1e-4; // Cutoff for sigmoid ramp 
    double steep = log(1./eps - 1.) * 2. / tramp; // steepness of sigmoid such that ramp(x) small for x < - tramp/2
    // printf("steep der %f\n", steep);

    double dramp_dtstop= 0.0;
    if (time <= tstart + tramp) { // ramp up
      dramp_dtstop = 0.0;
    }
    else if (tstart + tramp <= time && 
            time <= tstop - tramp) { // center
      dramp_dtstop = 0.0;
    }
    else if (time >= tstop - tramp && time <= tstop) { // down
      double center = tstop - tramp/2.0;
      // dramp_dtstop = sigmoid_diff(steep, -(time - center));
      // steep = 1842.048073;
      dramp_dtstop = 1.0/tramp;
    }
    
    // If ramp time is larger than total amount of time, turn off control:
    if (tstop < tstart + 2*tramp) dramp_dtstop= 0.0;

    return dramp_dtstop;
}

int getIndexReal(const int i) {
  return 2*i;
}

int getIndexImag(const int i) {
  return 2*i + 1;
}

int getVecID(const int row, const int col, const int dim){
  return row + col * dim;  
} 


int mapEssToFull(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential){

  int id = 0;
  int index = i;
  for (int iosc = 0; iosc<nlevels.size()-1; iosc++){
    int postdim = 1;
    int postdim_ess = 1;
    for (int j = iosc+1; j<nlevels.size(); j++){
      postdim *= nlevels[j];
      postdim_ess *= nessential[j];
    }
    int iblock = (int) index / postdim_ess;
    index = index % postdim_ess;
    // move id to that block
    id += iblock * postdim;  
  }
  // move to index in last block
  id += index;

  return id;
}

int mapFullToEss(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential){

  int id = 0;
  int index = i;
  for (int iosc = 0; iosc<nlevels.size(); iosc++){
    int postdim = 1;
    int postdim_ess = 1;
    for (int j = iosc+1; j<nlevels.size(); j++){
      postdim *= nlevels[j];
      postdim_ess *= nessential[j];
    }
    int iblock = (int) index / postdim;
    index = index % postdim;
    if (iblock >= nessential[iosc]) return -1; // this row/col belongs to a guard level, no mapping defined. 
    // move id to that block
    id += iblock * postdim_ess;  
  }

  return id;
}



// void projectToEss(Vec state,const std::vector<int> &nlevels, const std::vector<int> &nessential){

//   /* Get dimensions */
//   int dim_rho = 1;
//   for (int i=0; i<nlevels.size(); i++){
//     dim_rho *=nlevels[i];
//   }

//   /* Get local ownership of the state */
//   PetscInt ilow, iupp;
//   VecGetOwnershipRange(state, &ilow, &iupp);

//   /* Iterate over rows of system matrix, check if it corresponds to an essential level, and if not, set this row and colum to zero */
//   int reID, imID;
//   for (int i=0; i<dim_rho; i++) {
//     // zero out row and column if this does not belong to an essential level
//     if (!isEssential(i, nlevels, nessential)) { 
//       for (int j=0; j<dim_rho; j++) {
//         // zero out row
//         reID = getIndexReal(getVecID(i,j,dim_rho));
//         imID = getIndexImag(getVecID(i,j,dim_rho));
//         if (ilow <= reID && reID < iupp) VecSetValue(state, reID, 0.0, INSERT_VALUES);
//         if (ilow <= imID && imID < iupp) VecSetValue(state, imID, 0.0, INSERT_VALUES);
//         // zero out colum
//         reID = getIndexReal(getVecID(j,i,dim_rho));
//         imID = getIndexImag(getVecID(j,i,dim_rho));
//         if (ilow <= reID && reID < iupp) VecSetValue(state, reID, 0.0, INSERT_VALUES);
//         if (ilow <= imID && imID < iupp) VecSetValue(state, imID, 0.0, INSERT_VALUES);
//       }
//     } 
//   }
//   VecAssemblyBegin(state);
//   VecAssemblyEnd(state);


// }

int isEssential(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential) {

  int isEss = 1;
  int index = i;
  for (int iosc = 0; iosc < nlevels.size(); iosc++){

    int postdim = 1;
    for (int j = iosc+1; j<nlevels.size(); j++){
      postdim *= nlevels[j];
    }
    int itest = (int) index / postdim;
    // test if essential for this oscillator
    if (itest >= nessential[iosc]) {
      isEss = 0;
      break;
    }
    index = index % postdim;
  }

  return isEss; 
}

int isGuardLevel(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential){
  int isGuard =  0;
  int index = i;
  for (int iosc = 0; iosc < nlevels.size(); iosc++){

    int postdim = 1;
    for (int j = iosc+1; j<nlevels.size(); j++){
      postdim *= nlevels[j];
    }
    int itest = (int) index / postdim;   // floor(i/n_post)
    // test if this is a guard level for this oscillator
    if (itest == nlevels[iosc] - 1 && itest >= nessential[iosc]) {  // last energy level for system 'iosc'
      isGuard = 1;
      break;
    }
    index = index % postdim;
  }

  return isGuard;
}

PetscErrorCode Ikron(const Mat A,const  int dimI, const double alpha, Mat *Out, InsertMode insert_mode){

    PetscInt ierr;
    PetscInt ncols;
    const PetscInt* cols; 
    const PetscScalar* Avals;
    PetscInt* shiftcols;
    PetscScalar* vals;
    PetscInt dimA;
    PetscInt dimOut;
    PetscInt nonzeroOut;
    PetscInt rowID;

    MatGetSize(A, &dimA, NULL);

    ierr = PetscMalloc1(dimA, &shiftcols); CHKERRQ(ierr);
    ierr = PetscMalloc1(dimA, &vals); CHKERRQ(ierr);

    /* Loop over dimension of I */
    for (PetscInt i = 0; i < dimI; i++){

        /* Set the diagonal block (i*dimA)::(i+1)*dimA */
        for (PetscInt j=0; j<dimA; j++){
            MatGetRow(A, j, &ncols, &cols, &Avals);
            rowID = i*dimA + j;
            for (int k=0; k<ncols; k++){
                shiftcols[k] = cols[k] + i*dimA;
                vals[k] = Avals[k] * alpha;
            }
            MatSetValues(*Out, 1, &rowID, ncols, shiftcols, vals, insert_mode);
            MatRestoreRow(A, j, &ncols, &cols, &Avals);
        }

    }
    // MatAssemblyBegin(*Out, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(*Out, MAT_FINAL_ASSEMBLY);

    PetscFree(shiftcols);
    PetscFree(vals);
    return 0;
}

PetscErrorCode kronI(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode){
    
    PetscInt ierr;
    PetscInt dimA;
    const PetscInt* cols; 
    const PetscScalar* Avals;
    PetscInt rowid;
    PetscInt colid;
    PetscScalar insertval;
    PetscInt dimOut;
    PetscInt nonzeroOut;
    PetscInt ncols;
    MatInfo Ainfo;
    MatGetSize(A, &dimA, NULL);

    ierr = PetscMalloc1(dimA, &cols); CHKERRQ(ierr);
    ierr = PetscMalloc1(dimA, &Avals);

    /* Loop over rows in A */
    for (PetscInt i = 0; i < dimA; i++){
        MatGetRow(A, i, &ncols, &cols, &Avals);

        /* Loop over non negative columns in row i */
        for (PetscInt j = 0; j < ncols; j++){
            //printf("A: row = %d, col = %d, val = %f\n", i, cols[j], Avals[j]);
            
            // dimI rows. global row indices: i, i+dimI
            for (int k=0; k<dimI; k++) {
               rowid = i*dimI + k;
               colid = cols[j]*dimI + k;
               insertval = Avals[j] * alpha;
               MatSetValues(*Out, 1, &rowid, 1, &colid, &insertval, insert_mode);
              //  printf("Setting %d,%d %f\n", rowid, colid, insertval);
            }
        }
        MatRestoreRow(A, i, &ncols, &cols, &Avals);
    }

    // MatAssemblyBegin(*Out, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(*Out, MAT_FINAL_ASSEMBLY);

    PetscFree(cols);
    PetscFree(Avals);

    return 0;
}



PetscErrorCode AkronB(const Mat A, const Mat B, const double alpha, Mat *Out, InsertMode insert_mode){
    PetscInt Adim1, Adim2, Bdim1, Bdim2;
    MatGetSize(A, &Adim1, &Adim2);
    MatGetSize(B, &Bdim1, &Bdim2);

    PetscInt ncolsA, ncolsB;
    const PetscInt *colsA, *colsB;
    const double *valsA, *valsB;
    // Iterate over rows of A 
    for (PetscInt irowA = 0; irowA < Adim1; irowA++){
        // Iterate over non-zero columns in this row of A
        MatGetRow(A, irowA, &ncolsA, &colsA, &valsA);
        for (PetscInt j=0; j<ncolsA; j++) {
            PetscInt icolA = colsA[j];
            PetscScalar valA = valsA[j];
            /* put a B-block at position (irowA*Bdim1, icolA*Bdim2): */
            // Iterate over rows of B 
            for (PetscInt irowB = 0; irowB < Bdim1; irowB++){
                // Iterate over non-zero columns in this B-row
                MatGetRow(B, irowB, &ncolsB, &colsB, &valsB);
                for (PetscInt k=0; k< ncolsB; k++) {
                    PetscInt icolB = colsB[k];
                    PetscScalar valB = valsB[k];
                    /* Insert values in Out */
                    PetscInt rowOut = irowA*Bdim1 + irowB;
                    PetscInt colOut = icolA*Bdim2 + icolB;
                    PetscScalar valOut = valA * valB * alpha; 
                    MatSetValue(*Out, rowOut, colOut, valOut, insert_mode);
                }
                MatRestoreRow(B, irowB, &ncolsB, &colsB, &valsB);
            }
        }   
        MatRestoreRow(A, irowA, &ncolsA, &colsA, &valsA);
    }  

  return 0;
}


PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag) {
  
  int ierr; 

  /* Create B = -A */
  Mat B;
  ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &B); CHKERRQ(ierr);
  ierr = MatScale(B, -1.0); CHKERRQ(ierr);

  /* Test if B^T = A */
  ierr = MatIsTranspose(B, A, tol, flag); CHKERRQ(ierr);

  /* Cleanup */
  ierr = MatDestroy(&B); CHKERRQ(ierr);

  return ierr;
}



PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag) {
  int ierr;
  int i, j;

  /* TODO: Either make this work in Petsc-parallel, or add error exit if this runs in parallel. */
  
  /* Get u and v from x */
  PetscInt dim;
  ierr = VecGetSize(x, &dim); CHKERRQ(ierr);
  dim = dim/2;
  Vec u, v;
  IS isu, isv;

  int dimis = dim;
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 0, 2, &isu); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 1, 2, &isv); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isu, &u); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isv, &v); CHKERRQ(ierr);

  /* Init flags*/
  *flag = PETSC_TRUE;

  /* Check for symmetric u and antisymmetric v */
  const double *u_array;
  const double *v_array;
  double u_diff, v_diff;
  ierr = VecGetArrayRead(u, &u_array); CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &v_array); CHKERRQ(ierr);
  int N = sqrt(dim);
  for (i=0; i<N; i++) {
    for (j=i; j<N; j++) {
      u_diff = u_array[i*N+j] - u_array[j*N+i];
      v_diff = v_array[i*N+j] + v_array[j*N+i];
      if (fabs(u_diff) > tol || fabs(v_diff) > tol ) {
        *flag = PETSC_FALSE;
        break;
      }
    }
  }

  ierr = VecRestoreArrayRead(u, &u_array);
  ierr = VecRestoreArrayRead(v, &v_array);
  ierr = VecRestoreSubVector(x, isu, &u);
  ierr = VecRestoreSubVector(x, isv, &v);
  ISDestroy(&isu);
  ISDestroy(&isv);

  return ierr;
}



PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag) {

  int ierr;
  int i;

  /* Get u and v from x */
  PetscInt dim;
  ierr = VecGetSize(x, &dim); CHKERRQ(ierr);
  int dimis = dim/2;
  Vec u, v;
  IS isu, isv;
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 0, 2, &isu); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 1, 2, &isv); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isu, &u); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isv, &v); CHKERRQ(ierr);

  /* Init flags*/
  *flag = PETSC_FALSE;
  PetscBool u_hastrace1 = PETSC_FALSE;
  PetscBool v_hasdiag0  = PETSC_FALSE;

  /* Check if diagonal of u sums to 1, and diagonal elements of v are 0 */ 
  const double *u_array;
  const double *v_array;
  double u_sum = 0.0;
  double v_sum = 0.0;
  ierr = VecGetArrayRead(u, &u_array); CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &v_array); CHKERRQ(ierr);
  int N = sqrt(dimis);
  for (i=0; i<N; i++) {
    u_sum += u_array[i*N+i];
    v_sum += fabs(v_array[i*N+i]);
  }
  if ( fabs(u_sum - 1.0) < tol ) u_hastrace1 = PETSC_TRUE;
  if ( fabs(v_sum      ) < tol ) v_hasdiag0  = PETSC_TRUE;

  /* Restore vecs */
  ierr = VecRestoreArrayRead(u, &u_array);
  ierr = VecRestoreArrayRead(v, &v_array);
  ierr = VecRestoreSubVector(x, isu, &u);
  ierr = VecRestoreSubVector(x, isv, &v);

  /* Answer*/
  if (u_hastrace1 && v_hasdiag0) {
    *flag = PETSC_TRUE;
  }
  
  /* Destroy vector strides */
  ISDestroy(&isu);
  ISDestroy(&isv);


  return ierr;
}



PetscErrorCode SanityTests(Vec x, double time){

  /* Sanity check. Be careful: This is costly! */
  printf("Trace check %f ...\n", time);
  PetscBool check;
  double tol = 1e-10;
  StateIsHermitian(x, tol, &check);
  if (!check) {
    printf("WARNING at t=%f: rho is not hermitian!\n", time);
    printf("\n rho :\n");
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    exit(1);
  }
  else printf("IsHermitian check passed.\n");
  StateHasTrace1(x, tol, &check);
  if (!check) {
    printf("WARNING at t=%f: Tr(rho) is NOT one!\n", time);
    printf("\n rho :\n");
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    exit(1);
  }
  else printf("Trace1 check passed.\n");

  return 0;
}


int read_vector(const char *filename, double *var, int dim, bool quietmode, int skiplines, const std::string testheader) {

  FILE *file;
  double tmp;
  int success = 0;

  file = fopen(filename, "r");

  if (file != NULL) {
    if (!quietmode) printf("Reading file %s, starting from line %d.\n", filename, skiplines+1);

    /* Scip first <skiplines> lines */
    char buffer[51]; // need one extra element because fscanf adds a '\0' at the end
    for (int ix = 0; ix < skiplines; ix++) {
      fscanf(file, "%50[^\n]%*c", buffer); // // NOTE: &buffer[50] is a pointer to buffer[50] (i.e. its last element)
      // printf("Skipping %d lines: %s \n:", skiplines, buffer);
    }

    // Test the header, if given, and set vals to zero if header doesn't match.
    if (testheader.size()>0) {
      if (!quietmode) printf("Compare to Header '%s': ", testheader.c_str());
      // read one (first) line
      int ret = fscanf(file, "%50[^\n]%*c", buffer); // NOTE: &buffer[50] is a pointer to buffer[50] (i.e. its last element)
      std::string header = buffer;
      // Compare to testheader, return if it doesn't match 
      if (ret==EOF || header.compare(0,testheader.size(),testheader) != 0) {
        // printf("Header not found: %s != %s\n", header.c_str(), testheader.c_str());
        printf("Header not found. Got this instead: %s, skipped %d lines\n", header.c_str(), skiplines);
        for (int ix = 0; ix < dim; ix++) var[ix] = 0.0;
        fclose(file);
        return success;
      } else {
        if (!quietmode) printf(" Header correct! Reading now.\n");
      }
    }
 
    // printf("Either matching header, or no header given. Now reading lines \n");

    /* Read <dim> lines from file */
    for (int ix = 0; ix < dim; ix++) {
      double myval = 0.0;
      // read the next line
      int ret = fscanf(file, "%lf", &myval); 
      // if end of file, set remaining vars to zero
      if (ret == EOF){ 
        for (int j = ix; j<dim; j++) var[j] = 0.0;
        break;
      } else { // otherwise, set the value
        var[ix] = myval;
      }
      success = 1;
    }
  } else {
    printf("ERROR: Can't open file %s\n", filename);
    exit(1);
  }

  fclose(file);
  return success;
}


/* Compute eigenvalues */
int getEigvals(const Mat A, const int neigvals, std::vector<double>& eigvals, std::vector<Vec>& eigvecs){

int nconv = 0;
#ifdef WITH_SLEPC

  /* Create Slepc's eigensolver */
  EPS eigensolver;       
  EPSCreate(PETSC_COMM_WORLD, &eigensolver);
  EPSSetOperators(eigensolver, A, NULL);
  EPSSetProblemType(eigensolver, EPS_NHEP);
  EPSSetFromOptions(eigensolver);

  /* Number of requested eigenvalues */
  EPSSetDimensions(eigensolver,neigvals,PETSC_DEFAULT,PETSC_DEFAULT);

  // Solve eigenvalue problem
  int ierr = EPSSolve(eigensolver); CHKERRQ(ierr);

  /* Get information about convergence */
  int its, nev, maxit;
  EPSType type;
  double tol;
  EPSGetIterationNumber(eigensolver,&its);
  EPSGetType(eigensolver,&type);
  EPSGetDimensions(eigensolver,&nev,NULL,NULL);
  EPSGetTolerances(eigensolver,&tol,&maxit);
  EPSGetConverged(eigensolver, &nconv );

  PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);
  PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenpairs: %D\n",nev);
  PetscPrintf(PETSC_COMM_WORLD," Number of iterations taken: %D / %D\n",its, maxit);
  PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g\n",(double)tol);
  PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);

  /* Allocate eigenvectors */
  Vec eigvec;
  MatCreateVecs(A, &eigvec, NULL);

  // Get the result
  double kr, ki, error;
  // printf("Eigenvalues: \n");
  for (int j=0; j<nconv; j++) {
      EPSGetEigenpair( eigensolver, j, &kr, &ki, eigvec, NULL);
      EPSComputeError( eigensolver, j, EPS_ERROR_RELATIVE, &error );
      // printf("%f + i%f (err %f)\n", kr, ki, error);

      /* Store the eigenpair */
      eigvals.push_back(kr);
      eigvecs.push_back(eigvec);
      if (ki != 0.0) printf("Warning: eigenvalue imaginary! : %f", ki);
  }
  // printf("\n");
  // EPSView(eigensolver, PETSC_VIEWER_STDOUT_WORLD);

  /* Clean up*/
  EPSDestroy(&eigensolver);
#endif
  return nconv;
}

// test if A+iB is a unitary matrix: (A+iB)^\dag (A+iB) = I!
bool isUnitary(const Mat V_re, const Mat V_im){
  Mat C, D;
  double norm;
  bool isunitary = true;

  // test: C=V_re^T V_re + Vim^TVim should be the identity!
  MatTransposeMatMult(V_re, V_re, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  MatTransposeMatMult(V_im, V_im, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D);
  MatAXPY(C, 1.0, D, DIFFERENT_NONZERO_PATTERN); 
  MatShift(C, -1.0);
  MatNorm(C, NORM_FROBENIUS, &norm);
  if (norm > 1e-12) {
    printf("Unitary Test: V_re^TVre+Vim^TVim is not the identity! %1.14e\n", norm);
    // MatView(C, NULL);
    isunitary = false;
  } 
  MatDestroy(&C);
  MatDestroy(&D);

  // test: C=V_re^T V_im - Vre^TVim should be zero!
  MatTransposeMatMult(V_re, V_im, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
  MatTransposeMatMult(V_im, V_re, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D);
  MatAXPY(C, -1.0, D, DIFFERENT_NONZERO_PATTERN); 
  MatNorm(C, NORM_FROBENIUS, &norm);
  if (norm > 1e-12) {
    printf("Unitary Test: Vre^TVim - Vim^TVre is not zero! %1.14e\n", norm);
    // MatView(C,NULL);
    isunitary = false;
  }
  MatDestroy(&C);
  MatDestroy(&D);

  return isunitary;
}