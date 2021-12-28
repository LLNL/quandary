//Full ExaTN API: exatn_numerics.hpp


#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "mastereq.hpp"
#include <petscmat.h>

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <iostream>
#include <ios>
#include <utility>
#include <numeric>
#include <chrono>
#include <thread>

using namespace exatn;

int main(int argc, char ** argv)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  
  /* Read config file */
  if (argc < 2) {
   printf("\nUSAGE: ./main </path/to/configfile> \n");
   exit(1);
   return 0;
  }
  std::stringstream log;
  MapParam config(MPI_COMM_WORLD, log);
  config.ReadFile(argv[1]);

  /* --- Get stuff from config file --- */
  std::vector<int> nlevels;
  config.GetVecIntParam("nlevels", nlevels, 0);
  int ntime = config.GetIntParam("ntime", 1000);
  int nspline = config.GetIntParam("nspline", 10);
  double dt    = config.GetDoubleParam("dt", 0.01);
  double total_time = ntime * dt;
  std::vector<int> nessential(nlevels.size());
  for (int iosc = 0; iosc<nlevels.size(); iosc++) nessential[iosc] = nlevels[iosc];
  /* Overwrite if config option is given */
  std::vector<int> read_nessential;
  config.GetVecIntParam("nessential", read_nessential, -1);
  if (read_nessential[0] > -1) {
    for (int iosc = 0; iosc<nlevels.size(); iosc++){
      if (iosc < read_nessential.size()) nessential[iosc] = read_nessential[iosc];
      else                               nessential[iosc] = read_nessential[read_nessential.size()-1];
    }
  }
  /* --- Initialize the Oscillators --- */
  Oscillator** oscil_vec = new Oscillator*[nlevels.size()];
  // Get fundamental and rotation frequencies from config file
  std::vector<double> trans_freq, rot_freq;
  config.GetVecDoubleParam("transfreq", trans_freq, 1e20);
  if (trans_freq.size() < nlevels.size()) {
    printf("Error: Number of given fundamental frequencies (%lu) is smaller than the the number of oscillators (%lu)\n", trans_freq.size(), nlevels.size());
    exit(1);
  }
  config.GetVecDoubleParam("rotfreq", rot_freq, 1e20);
  if (rot_freq.size() < nlevels.size()) {
    printf("Error: Number of given rotation frequencies (%lu) is smaller than the the number of oscillators (%lu)\n", rot_freq.size(), nlevels.size());
    exit(1);
  }
  // Get self kerr coefficient
  std::vector<double> selfkerr;
  config.GetVecDoubleParam("selfkerr", selfkerr, 0.0);   // self ker \xi_k
  assert(selfkerr.size() >= nlevels.size());
  // Get lindblad type and collapse times
  std::string lindblad = config.GetStrParam("collapse_type", "none");
  std::vector<double> decay_time, dephase_time;
  config.GetVecDoubleParam("decay_time", decay_time, 0.0);
  config.GetVecDoubleParam("dephase_time", dephase_time, 0.0);
  LindbladType lindbladtype;
  if      (lindblad.compare("none")      == 0 ) lindbladtype = LindbladType::NONE;
  else if (lindblad.compare("decay")     == 0 ) lindbladtype = LindbladType::DECAY;
  else if (lindblad.compare("dephase")   == 0 ) lindbladtype = LindbladType::DEPHASE;
  else if (lindblad.compare("both")      == 0 ) lindbladtype = LindbladType::BOTH;
  else {
    printf("\n\n ERROR: Unnown lindblad type: %s.\n", lindblad.c_str());
    printf(" Choose either 'none', 'decay', 'dephase', or 'both'\n");
    exit(1);
  }
  if (lindbladtype != LindbladType::NONE) {
    assert(decay_time.size() >= nlevels.size());
    assert(dephase_time.size() >= nlevels.size());
  }

  // Create the oscillators
  for (int i = 0; i < nlevels.size(); i++){
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(i);
    config.GetVecDoubleParam(key, carrier_freq, 0.0);
    oscil_vec[i] = new Oscillator(i, nlevels, nspline, trans_freq[i], selfkerr[i], rot_freq[i], decay_time[i], dephase_time[i], carrier_freq, total_time);
  }



  /* --- Initialize the Master Equation  --- */
  // Get self and cross kers and coupling terms 
  std::vector<double> crosskerr, Jkl;
  config.GetVecDoubleParam("crosskerr", crosskerr, 0.0);   // cross ker \xi_{kl}, zz-coupling
  config.GetVecDoubleParam("Jkl", Jkl, 0.0); // Jaynes-Cummings coupling
  // If not enough elements are given, fill up with zeros!
  int noscillators = nlevels.size();
  for (int i = crosskerr.size(); i < (noscillators-1) * noscillators / 2; i++)  crosskerr.push_back(0.0);
  for (int i = Jkl.size(); i < (noscillators-1) * noscillators / 2; i++) Jkl.push_back(0.0);
  // Sanity check for matrix free solver
  bool usematfree = config.GetBoolParam("usematfree", false);
  if ( (usematfree && nlevels.size() < 2) ||   
       (usematfree && nlevels.size() > 5)   ){
        printf("Warning: Matrix free solver is only implemented for systems with 2, 3, 4, or 5 oscillators. Switching to sparse-matrix solver now.\n");
        usematfree = false;
  }
  // Compute coupling rotation frequencies eta_ij = w^r_i - w^r_j
  std::vector<double> eta(nlevels.size()*(nlevels.size()-1)/2.);
  int idx = 0;
  for (int iosc=0; iosc<nlevels.size(); iosc++){
    for (int josc=iosc+1; josc<nlevels.size(); josc++){
      eta[idx] = rot_freq[iosc] - rot_freq[josc];
      idx++;
    }
  }

  MasterEq* mastereq = new MasterEq(nlevels, nessential, oscil_vec, crosskerr, Jkl, eta, lindbladtype, usematfree);



  /* Create rho_in */
  Vec vec_rho_in;
  VecCreate(PETSC_COMM_WORLD, &vec_rho_in);
  VecSetSizes(vec_rho_in,PETSC_DECIDE,2*mastereq->getDim());
  VecSetFromOptions(vec_rho_in);
  for (int i=0; i<mastereq->getDim(); i++){
    VecSetValue(vec_rho_in, getIndexReal(i),  1.0*i +1.0, INSERT_VALUES);
    VecSetValue(vec_rho_in, getIndexImag(i),  0.0, INSERT_VALUES);
  }
  //VecView(vec_rho_in, NULL);
  // Initialize rho_out
  Vec vec_rho_out;
  VecDuplicate(vec_rho_in, &vec_rho_out);

  /* Evaluate M(0) */
  Mat RHS = mastereq->getRHS();
  mastereq->assemble_RHS(0.0);
  Mat M = mastereq->getRHS();


  /* matmult y = Mx */
  MatMult(M, vec_rho_in, vec_rho_out);

  //std::cout<<" ######## PETSC result ########" << std::endl;
  //VecView(vec_rho_out, NULL);


  /*----- EXATN ----- */
 // Some ExaTN stuff
 exatn::ParamConf exatn_parameters;
 //Set the available CPU Host RAM size to be used by ExaTN:
 exatn_parameters.setParameter("host_memory_buffer_size",4L*1024L*1024L*1024L);
#ifdef MPI_ENABLED
 int thread_provided;
 int mpi_error = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_provided);
 assert(mpi_error == MPI_SUCCESS);
 assert(thread_provided == MPI_THREAD_MULTIPLE);
 exatn::initialize(exatn::MPICommProxy(MPI_COMM_WORLD),exatn_parameters,"lazy-dag-executor");
#else
 exatn::initialize(exatn_parameters,"lazy-dag-executor");
#endif

 const auto TENS_ELEM_TYPE = TensorElementType::REAL64;
 //const auto TENS_ELEM_TYPE = TensorElementType::COMPLEX64;

 { // scope for exatn
  auto success = true;


  /* Input and output density matrix */
  // Declare tensors
  std::vector<std::size_t> dm_rho; 
  for (int i=0; i<nlevels.size(); i++) dm_rho.push_back(nlevels[i]);
  for (int i=0; i<nlevels.size(); i++) dm_rho.push_back(nlevels[i]);  // do it two times because rho \in C^{prod_i n_i} x {prod_i n_i}, so once for each dimension of the matrix rho.
  //for (int i=0; i<dm_rho.size(); i++) std::cout<< dm_rho[i] << "  " << std::endl;
  auto rho_in_re = exatn::makeSharedTensor("RhoInRe",TensorShape(dm_rho));
  auto rho_in_im = exatn::makeSharedTensor("RhoInIm",TensorShape(dm_rho));
  auto rho_out_re= exatn::makeSharedTensor("RhoOutRe",TensorShape(dm_rho));
  auto rho_out_im= exatn::makeSharedTensor("RhoOutIm",TensorShape(dm_rho));
  // Create tensors
  success = exatn::createTensor(rho_in_re,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_in_im,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_out_re,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_out_im,TENS_ELEM_TYPE); assert(success);
  // Initialize tensors
  success = exatn::initTensor("RhoOutRe",0.0); assert(success);
  success = exatn::initTensor("RhoOutIm",0.0); assert(success); 
  success = exatn::initTensor("RhoInIm",0.0); assert(success); 
  // Input vector
  std::vector<double> rho_in_data;
  for (int i=0; i<mastereq->getDim(); i++){ // dim = N^2
    rho_in_data.push_back(1.0*i+1.0);
  }
  success = exatn::initTensorData("RhoInRe",rho_in_data); assert(success);

  // Print Tensor
  //std::cout<<"Exa input: " << std::endl;
  //exatn::printTensor("RhoInRe");

  // Dimensions
  long unsigned int n0 = nlevels[0];
  long unsigned int n1 = nlevels[1];
  std::vector<std::size_t> dim_q{n0,n0}; // TODO: Generalize for oscillators with different numbers of levels

  /* Operators */
  // Number operator, same size for all oscillators
  auto numberOP = makeSharedTensor("NumberOP", TensorShape(dim_q));
  success = createTensor(numberOP, TensorElementType::REAL64); assert(success);
  success = initTensor("NumberOP", 0.0); assert(success);
  success = initTensorData("NumberOP", std::vector<double>{0.0, 0.0, 0.0, 1.0}); assert(success);
  //Lowering operator
  auto lowering = makeSharedTensor("Lowering", TensorShape(dim_q));
  success = createTensor(lowering, TensorElementType::REAL64); assert(success);
  success = initTensor("Lowering", 0.0); assert(success);
  success = initTensorData("Lowering", std::vector<double>{0.0, 0.0, 1.0, 0.0}); assert(success);
  //printTensor("Lowering");

  /* --- Hamiltonian -i(Hrho - rhoH) --- */

  /* Detuning */
  double omega0 = oscil_vec[0]->detuning_freq;
  double omega1 = oscil_vec[1]->detuning_freq;
  printf("detuning %f %f\n", omega0, omega1);
  
  // real part
  success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,j2,i3,i4)*NumberOP(i2,j2)",+omega0); assert(success); // 1st qubit left: + H\rho
  success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(j1,i2,i3,i4)*NumberOP(i1,j1)",+omega1); assert(success); // 2nd qubit left: + H\rho
  success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,i2,i3,j4)*NumberOP(j4,i4)",-omega0); assert(success); // 1st qubit right - \rho H
  success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,i2,j3,i4)*NumberOP(j3,i3)",-omega1); assert(success); // 2nd qubit right - \rho H
  // imag part
  success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,j2,i3,i4)*NumberOP(i2,j2)",-omega0); assert(success); // 1st qubit left: - H\rho
  success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(j1,i2,i3,i4)*NumberOP(i1,j1)",-omega1); assert(success); // 2nd qubit left: - H\rho
  success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,i2,i3,j4)*NumberOP(j4,i4)",+omega0); assert(success); // 1st qubit right: +\rho H
  success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,i2,j3,i4)*NumberOP(j3,i3)",+omega1); assert(success); // 2nd qubit right: +\rho H

  success = exatn::sync(); assert(success);


  //std::cout<<" ######## ExaTN result ########" << std::endl;
  //printTensor("RhoOutRe");
  //printTensor("RhoOutIm");


  
  /*****  Compare Petsc and ExaTn results */
  char tensornameRe[] = "RhoOutRe";
  char tensornameIm[] = "RhoOutIm";
  auto local_copy_Re = exatn::getLocalTensor(tensornameRe); assert(local_copy_Re); //type = talsh::Tensor
  auto local_copy_Im = exatn::getLocalTensor(tensornameIm); assert(local_copy_Im); //type = talsh::Tensor
  auto tensor_view_Re = local_copy_Re->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //full tensor view
  auto tensor_view_Im = local_copy_Im->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //full tensor view

  double err = 0.0;
  for (int i0=0; i0<nlevels[0]; i0++){
    for (int i1=0; i1<nlevels[1]; i1++){
      for (int i0p=0; i0p<nlevels[0]; i0p++){
        for (int i1p=0; i1p<nlevels[1]; i1p++){
            // ExaTN values
            //std::cout << "RhoIm["<< i0 << "," << i1 << "," << i0p << "," << i1p << "] = " << tensor_view[{i0,i1, i0p, i1p}] << std::endl;
            double valRe_exaTN = tensor_view_Re[{i0,i1, i0p, i1p}];
            double valIm_exaTN = tensor_view_Im[{i0,i1, i0p, i1p}];

            // PETSC values
            int i = i0 + i1 * nlevels[1];
            int j = i0p + i1p * nlevels[1];
            int vecID = getVecID(i,j,nlevels[0]*nlevels[1]);
            int vecID_re = getIndexReal(vecID);
            int vecID_im = getIndexImag(vecID);
            double valRe_Petsc = 0.0;
            double valIm_Petsc = 0.0;
            VecGetValues(vec_rho_out, 1, &vecID_re, &valRe_Petsc);
            VecGetValues(vec_rho_out, 1, &vecID_im, &valIm_Petsc);
            //std::cout << "PetscIm = " << val_Petsc << std::endl;
            
            // error
            err += pow(valRe_exaTN - valRe_Petsc, 2.0);
            err += pow(valIm_exaTN - valIm_Petsc, 2.0);
        }
      }
    }
  }
  err = sqrt(err);
  std::cout<< std::endl << "Error(PetscVsExaTN) = " << err << std::endl << std::endl;

  local_copy_Re.reset();
  local_copy_Im.reset();

 /*******************************/
 /* EXATN TEMPLATES */
  /* apply lowering operation */
  // LEFT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(i1,j2,i3,i4)*Lowering(i2,j2)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(j1,i2,i3,i4)*Lowering(i1,j1)",1.0); assert(success); // 2nd qubit
  // RIGHT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(i1,i2,i3,j4)*Lowering(j4,i4)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(i1,i2,j3,i4)*Lowering(j3,i3)",1.0); assert(success); // 2nd qubit

  /* apply raising operation */
  // LEFT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(i1,j2,i3,i4)*Lowering(j2,i2)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(j1,i2,i3,i4)*Lowering(j1,i1)",1.0); assert(success); // 2nd qubit
  // RIGHT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(i1,i2,i3,j4)*Lowering(i4,j4)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)+=RhoIn(i1,i2,j3,i4)*Lowering(i3,j3)",1.0); assert(success); // 2nd qubit
  
  //printTensor("RhoOut");


  ////Declare tensors:
  //std::vector<std::size_t> dm_extents(4,2);
  //auto density_matrix = exatn::makeSharedTensor("DensityMatrix",TensorShape(dm_extents));
  //auto result_matrix = exatn::makeSharedTensor("ResultMatrix",TensorShape(dm_extents));
  //std::vector<std::size_t> dim_q(2,2);
  //auto lowering = exatn::makeSharedTensor("Lowering",TensorShape(dim_q));
  //auto raising = exatn::makeSharedTensor("Raising",TensorShape(dim_q));

  ////Create tensors:
  //success = exatn::createTensor(density_matrix,TENS_ELEM_TYPE); assert(success);
  //success = exatn::createTensor(result_matrix,TENS_ELEM_TYPE); assert(success);
  //success = exatn::createTensor(lowering,TENS_ELEM_TYPE); assert(success);
  //success = exatn::createTensor(raising,TENS_ELEM_TYPE); assert(success);

  ////Initialize tensors:
  //success = exatn::initTensorRnd("DensityMatrix"); assert(success);  // random 
  //success = exatn::initTensor("ResultMatrix",0.0); assert(success);  // zeros
  //success = exatn::initTensorData("Lowering",                        // data: column-major! 
  //                                std::vector<double>{0.0, 1.0, 0.0, 0.0}); assert(success);
  //success = exatn::initTensorData("Raising",
  //                                std::vector<double>{0.0, 0.0, 1.0, 0.0}); assert(success);

  /* apply lowering operation */
  // LEFT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,j2,i3,i4)*Lowering(i2,j2)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(j1,i2,i3,i4)*Lowering(i1,j1)",1.0); assert(success); // 2nd qubit
  // RIGHT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,i2,i3,j4)*Lowering(j4,i4)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,i2,j3,i4)*Lowering(j3,i3)",1.0); assert(success); // 2nd qubit

  /* apply raising operation */
  // LEFT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,j2,i3,i4)*Lowering(j2,i2)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(j1,i2,i3,i4)*Lowering(j1,i1)",1.0); assert(success); // 2nd qubit
  // RIGHT
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,i2,i3,j4)*Lowering(i4,j4)",1.0); assert(success); // 1st qubit
  //success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,i2,j3,i4)*Lowering(i3,j3)",1.0); assert(success); // 2nd qubit
  


  ////Apply raising to the 2nd ket qubit in the density matrix:
  //success = exatn::contractTensors("ResultMatrix(i1,i2,i3,i4)+=DensityMatrix(i1,j1,i3,i4)*Raising(j1,i2)",1.0); assert(success);
  //  
  ////Synchronize execution: (WHEN?)
  //success = exatn::sync(); assert(success);

  ////Print tensor:
  //exatn::printTensor("ResultMatrix");

#if 0
  //Add tensors:
  success = exatn::addTensors("ResultMatrix(i1,i2,i3,i4)+=DensityMatrix(i1,i2,i3,i4)",1.0); assert(success);

  //Get access to data:
  auto local_copy = exatn::getLocalTensor("ResultMatrix"); assert(local_copy); //type = talsh::Tensor
  auto tensor_view = local_copy->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //full tensor view
#endif
 }

 exatn::finalize();
#ifdef MPI_ENABLED
 mpi_error = MPI_Finalize(); assert(mpi_error == MPI_SUCCESS);
#endif
 return 0;
}

