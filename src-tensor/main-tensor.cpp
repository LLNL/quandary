//Full ExaTN API: exatn_numerics.hpp


#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"
#include "config.hpp"

#ifdef MPI_ENABLED
#include "mpi.h"
#endif

#include <iostream>
#include <ios>
#include <utility>
#include <numeric>
#include <chrono>
#include <thread>

int main(int argc, char ** argv)
{
 using exatn::Tensor;
 using exatn::TensorShape;
 using exatn::TensorSignature;
 using exatn::TensorNetwork;
 using exatn::TensorOperator;
 using exatn::TensorExpansion;
 using exatn::TensorElementType;

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

 {
  auto success = true;

  //Declare tensors:
  std::vector<std::size_t> dm_extents(4,2);
  auto density_matrix = exatn::makeSharedTensor("DensityMatrix",TensorShape(dm_extents));
  auto result_matrix = exatn::makeSharedTensor("ResultMatrix",TensorShape(dm_extents));
  std::vector<std::size_t> dim_q(2,2);
  auto lowering = exatn::makeSharedTensor("Lowering",TensorShape(dim_q));
  auto raising = exatn::makeSharedTensor("Raising",TensorShape(dim_q));

  //Create tensors:
  success = exatn::createTensor(density_matrix,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(result_matrix,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(lowering,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(raising,TENS_ELEM_TYPE); assert(success);

  //Initialize tensors:
  success = exatn::initTensorRnd("DensityMatrix"); assert(success);  // random 
  success = exatn::initTensor("ResultMatrix",0.0); assert(success);  // zeros
  success = exatn::initTensorData("Lowering",                        // data: column-major! 
                                  std::vector<double>{0.0, 1.0, 0.0, 0.0}); assert(success);
  success = exatn::initTensorData("Raising",
                                  std::vector<double>{0.0, 0.0, 1.0, 0.0}); assert(success);

  //Apply raising to the 2nd ket qubit in the density matrix:
  success = exatn::contractTensors("ResultMatrix(i1,i2,i3,i4)+=DensityMatrix(i1,j1,i3,i4)*Raising(j1,i2)",1.0); assert(success);
    
  //Synchronize execution: (WHEN?)
  success = exatn::sync(); assert(success);

  //Print tensor:
  exatn::printTensor("ResultMatrix");

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

