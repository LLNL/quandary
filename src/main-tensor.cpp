//Full ExaTN API: exatn_numerics.hpp


#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"
#include "config.hpp"
#include "defs.hpp"

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

  ///* Read config file */
  //if (argc < 2) {
  // printf("\nUSAGE: ./main </path/to/configfile> \n");
  // exit(1);
  // return 0;
  //}
  //std::stringstream log;
  //MapParam config(MPI_COMM_WORLD, log);
  //config.ReadFile(argv[1]);

  ///* --- Get number of levels from the config file --- */
  //std::vector<int> nlevels;
  //config.GetVecIntParam("nlevels", nlevels, 0);
  //// dimension of Hilbertspace
  //int dim_H = 1;
  //for (int i=0; i<nlevels.size(); i++){
  //  dim_H *= nlevels[i];
  //}
  //assert (dim_H == 4);

 { // scope for exatn
  auto success = true;

  /* Input and output density matrix */
  // Declare tensors
  //std::vector<std::size_t> dm_rho; 
  //for (int i=0; i<nlevels.size(); i++) dm_rho.push_back(nlevels[i]);
  //for (int i=0; i<nlevels.size(); i++) dm_rho.push_back(nlevels[i]);  // do it two times because rho \in C^{prod_i n_i} x {prod_i n_i}, so once for each dimension of the matrix rho.
  std::vector<std::size_t> dm_rho(4,2); 
  auto rho_in = exatn::makeSharedTensor("RhoIn",TensorShape(dm_rho));
  auto rho_out= exatn::makeSharedTensor("RhoOut",TensorShape(dm_rho));
  // Create tensors
  success = exatn::createTensor(rho_in,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_out,TENS_ELEM_TYPE); assert(success);
  // Initialize tensors
  success = exatn::initTensor("RhoOut",0.0); assert(success); // Initialize output with zero
  std::vector<double> rho_in_data;
  for (int i=0; i<16; i++){
    rho_in_data.push_back(1.0*i+1.0);
  }
  success = exatn::initTensorData("RhoIn",rho_in_data); assert(success); // Initialize input with data.
  //success = exatn::initTensor("RhoIn",22.0); assert(success); 

  // Print Tensor
  std::cout<<"Input: " << std::endl;
  exatn::printTensor("RhoIn");


  /* Operators */
  // Number operator
  std::vector<std::size_t> dim_q(2,2);
  auto numberOP = makeSharedTensor("NumberOP", TensorShape(dim_q));
  success = createTensor(numberOP, TensorElementType::REAL64); assert(success);
  success = initTensor("NumberOP", 0.0); assert(success);
  success = initTensorData("NumberOP", std::vector<double>{0.0, 0.0, 0.0, 1.0}); assert(success);
  //Lowering operator
  auto lowering = makeSharedTensor("Lowering", TensorShape(dim_q));
  success = createTensor(lowering, TensorElementType::REAL64); assert(success);
  success = initTensor("Lowering", 0.0); assert(success);
  success = initTensorData("Lowering", std::vector<double>{0.0, 0.0, 1.0, 0.0}); assert(success);

  printTensor("Lowering");

  double omega0 = 5.0;
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
  success = exatn::contractTensors("RhoOut(i1,i2,i3,i4)=RhoIn(i1,i2,j3,i4)*Lowering(i3,j3)",1.0); assert(success); // 2nd qubit
  
  //printTensor("RhoOut");

  
  success = exatn::sync(); assert(success);

  std::cout<<"Output:" << std::endl;
  printTensor("RhoOut");


 /* TEMPLATES */

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

