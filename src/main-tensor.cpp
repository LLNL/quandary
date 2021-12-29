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


  // Initial rho data (input)
  std::vector<double> rho_in_re_data;
  std::vector<double> rho_in_im_data;
  for (int i=0; i<mastereq->getDim(); i++){ 
    rho_in_re_data.push_back(1.0*i+1.0);
    rho_in_im_data.push_back(10.0*i+10.0);
  }

  /* Create Petsc Vectors */
  Vec rho_in_petsc;
  Vec rho_out_petsc;
  VecCreate(PETSC_COMM_WORLD, &rho_in_petsc);
  VecSetSizes(rho_in_petsc,PETSC_DECIDE,2*mastereq->getDim());
  VecSetFromOptions(rho_in_petsc);
  for (int i=0; i<mastereq->getDim(); i++){
    VecSetValue(rho_in_petsc, getIndexReal(i),  rho_in_re_data[i], INSERT_VALUES);
    VecSetValue(rho_in_petsc, getIndexImag(i),  rho_in_im_data[i], INSERT_VALUES);
  }
  VecDuplicate(rho_in_petsc, &rho_out_petsc);

  /* Evaluate M(0) */
  Mat RHS = mastereq->getRHS();
  mastereq->assemble_RHS(0.0);
  Mat M = mastereq->getRHS();




  // Declare ExaTN tensors
  std::vector<std::size_t> dm_rho; 
  for (int i=0; i<nlevels.size(); i++) dm_rho.push_back(nlevels[i]);
  for (int i=0; i<nlevels.size(); i++) dm_rho.push_back(nlevels[i]);  // do it two times because rho \in C^{prod_i n_i} x {prod_i n_i}, so once for each dimension of the matrix rho.
  //for (int i=0; i<dm_rho.size(); i++) std::cout<< dm_rho[i] << "  " << std::endl;
  auto rho_in_re = exatn::makeSharedTensor("RhoInRe",TensorShape(dm_rho));
  auto rho_in_im = exatn::makeSharedTensor("RhoInIm",TensorShape(dm_rho));
  auto rho_out_re= exatn::makeSharedTensor("RhoOutRe",TensorShape(dm_rho));
  auto rho_out_im= exatn::makeSharedTensor("RhoOutIm",TensorShape(dm_rho));
  auto rho_aux = exatn::makeSharedTensor("RhoAux",TensorShape(dm_rho));
  // Create tensors
  success = exatn::createTensor(rho_in_re,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_in_im,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_out_re,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_out_im,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(rho_aux,TENS_ELEM_TYPE); assert(success);
  // Initialize tensors
  success = exatn::initTensor("RhoOutRe",0.0); assert(success);
  success = exatn::initTensor("RhoOutIm",0.0); assert(success); 
  success = exatn::initTensor("RhoAux",0.0); assert(success); 
  success = exatn::initTensorData("RhoInRe",rho_in_re_data); assert(success);   
  success = exatn::initTensorData("RhoInIm",rho_in_im_data); assert(success);  

  // Print Tensor
  //std::cout<<"Exa input: " << std::endl;
  //exatn::printTensor("RhoInRe");

  // Dimensions
  long unsigned int n0 = nlevels[0];
  long unsigned int n1 = nlevels[1];
  std::vector<std::size_t> dim_q{n0,n0}; // TODO: Generalize for oscillators with different numbers of levels

  /* Operators */
  std::vector<double> op_data;
 
  // Number operator, same size for all oscillators
  auto numberOP = makeSharedTensor("NumberOP", TensorShape(dim_q));
  success = createTensor(numberOP, TensorElementType::REAL64); assert(success);
  success = initTensor("NumberOP", 0.0); assert(success);
  op_data.clear();
  for (int i=0; i<n0; i++){
    for (int j=0; j<n0; j++){
      double val = 0.0;
      if (i == j) val = i;
      op_data.push_back(val);
    }
  }
  success = initTensorData("NumberOP", op_data); assert(success);
  //printTensor("NumberOP");
  
  // Selfkerr operator, same size for all oscillators
  auto selfkerrOP = makeSharedTensor("SelfKerrOP", TensorShape(dim_q));
  success = createTensor(selfkerrOP, TensorElementType::REAL64); assert(success);
  success = initTensor("SelfKerrOP", 0.0); assert(success);
  op_data.clear();
  for (int i=0; i<n0; i++){
    for (int j=0; j<n0; j++){
      double val = 0.0;
      if (i == j) val = i*i - i;
      op_data.push_back(val);
    }
  }
  success = initTensorData("SelfKerrOP", op_data); assert(success);
  //printTensor("SelfKerrOP");
 
  
  //Lowering operator
  auto loweringOP = makeSharedTensor("LoweringOP", TensorShape(dim_q));
  success = createTensor(loweringOP, TensorElementType::REAL64); assert(success);
  success = initTensor("LoweringOP", 0.0); assert(success);
  op_data.clear();
  for (int i=0; i<n0; i++){
    for (int j=0; j<n0; j++){
      double val = 0.0;
      if (i == j+1) val = sqrt(i);
      op_data.push_back(val);
    }
  }  
  //success = initTensorData("LoweringOP", std::vector<double>{0.0, 0.0, 1.0, 0.0}); assert(success);
  success = initTensorData("LoweringOP", op_data); assert(success);
  //printTensor("LoweringOP");



  // Run <nexe> matrix-vector multiplications
  int nexec = 20;

  /* ----------------------------------------------------------------------------------*/
  /* --- PETSC Matrix vector multiplication for system Hamiltonian -i(Hrho - rhoH) --- */
  /* ----------------------------------------------------------------------------------*/

  /* Start timer */
  double TimePetscStart = MPI_Wtime();
  double Timecurr = TimePetscStart;

  std::cout<<"Petsc: ";
  for (int iexec = 0; iexec < nexec; iexec++){
    std::cout<<" " << iexec;

    /* Petsc matrix vector product y = Mx */
    MatMult(M, rho_in_petsc, rho_out_petsc);


    std::cout<< "("<< MPI_Wtime() - Timecurr << " sec)";
    Timecurr = MPI_Wtime();
  }
  std::cout<<std::endl;


  //std::cout<<" ######## PETSC result ########" << std::endl;
  //VecView(rho_out_petsc, NULL);

  /* Get time */
  double TimePetscStop = MPI_Wtime();




  /* ------------------------------------------------------------------*/
  /* --- ExaTN Contractions for system Hamiltonian -i(Hrho - rhoH) --- */
  /* ------------------------------------------------------------------*/
  double omega0 = oscil_vec[0]->getDetuning();
  double omega1 = oscil_vec[1]->getDetuning();
  //printf("detuning %f %f\n", omega0, omega1);
  double xi0 = oscil_vec[0]->getSelfkerr();
  double xi1 = oscil_vec[1]->getSelfkerr();
  //printf("selfkerr %f %f\n", xi0, xi1);
  double xi01 = mastereq->getCrossKerr()[0];
  //printf("crosskerr %f\n", xi01);

  /* Start timer */
  double TimeExaStart = MPI_Wtime();
  Timecurr = TimeExaStart;

  std::cout<<"ExaTN: " ;
  for (int iexec = 0; iexec < nexec; iexec++){
    std::cout<<" " << iexec;
    success = exatn::initTensor("RhoOutRe",0.0); assert(success); // reset output
    success = exatn::initTensor("RhoOutIm",0.0); assert(success); // reset output

    /* Detuning  H = omega0 N \kron Id + omega1 Id \kron N */
    // first term, apply N to first qubit from left and right 
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,j2,i3,i4)*NumberOP(i2,j2)",+omega0); assert(success); // left:  + H \rho  (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,j2,i3,i4)*NumberOP(i2,j2)",-omega0); assert(success); // left:  - H \rho  (im)
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,i2,i3,j4)*NumberOP(j4,i4)",-omega0); assert(success); // right: - \rho H  (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,i2,i3,j4)*NumberOP(j4,i4)",+omega0); assert(success); // right: + \rho H  (im)
    // second term
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(j1,i2,i3,i4)*NumberOP(i1,j1)",+omega1); assert(success); // left:  + H \rho  (re) 
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(j1,i2,i3,i4)*NumberOP(i1,j1)",-omega1); assert(success); // left:  - H \rho  (im)
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,i2,j3,i4)*NumberOP(j3,i3)",-omega1); assert(success); // right: - \rho H  (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,i2,j3,i4)*NumberOP(j3,i3)",+omega1); assert(success); // right: + \rho H  (im)

    /* SelfKerr H = -xi0/2 (N^2-N) \kron Id - xi1/2 Id \kron (N^2 - N) */
    // first term: apply N^2-N to first qubit from left and right
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,j2,i3,i4)*SelfKerrOP(i2,j2)",-xi0/2.); assert(success); // left:  - H \rho (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,j2,i3,i4)*SelfKerrOP(i2,j2)",+xi0/2.); assert(success); // left:  + H \rho (im)
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,i2,i3,j4)*SelfKerrOP(j4,i4)",+xi0/2.); assert(success); // right: + \rho H (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,i2,i3,j4)*SelfKerrOP(j4,i4)",-xi0/2.); assert(success); // right: - \rho H (im)
    // second term 
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(j1,i2,i3,i4)*SelfKerrOP(i1,j1)",-xi1/2.); assert(success); // left:  - H \rho (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(j1,i2,i3,i4)*SelfKerrOP(i1,j1)",+xi1/2.); assert(success); // left:  + H \rho (im)
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoInIm(i1,i2,j3,i4)*SelfKerrOP(j3,i3)",+xi1/2.); assert(success); // right: + \rho H (re)
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoInRe(i1,i2,j3,i4)*SelfKerrOP(j3,i3)",-xi1/2.); assert(success); // right: - \rho H (im)

    /* CrossKerr coupling 0<->1 H = -xi01 N \kron N */
    // real part
    // left
    success = exatn::initTensor("RhoAux",0.0); assert(success); // reset rho_aux
    success = exatn::contractTensors("RhoAux(i1,i2,i3,i4)+=RhoInIm(j1,i2,i3,i4)*NumberOP(i1,j1)",1.0); assert(success); // aux = N rho
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoAux(i1,j2,i3,i4)*NumberOP(i2,j2)",-xi01); assert(success); // y += N aux
    // right 
    success = exatn::initTensor("RhoAux",0.0); assert(success); // reset rho_aux
    success = exatn::contractTensors("RhoAux(i1,i2,i3,i4)+=RhoInIm(i1,i2,j3,i4)*NumberOP(j3,i3)",1.0); assert(success); 
    success = exatn::contractTensors("RhoOutRe(i1,i2,i3,i4)+=RhoAux(i1,i2,i3,j4)*NumberOP(j4,i4)",+xi01); assert(success);
    // imag part
    // left
    success = exatn::initTensor("RhoAux",0.0); assert(success); // reset rho_aux
    success = exatn::contractTensors("RhoAux(i1,i2,i3,i4)+=RhoInRe(j1,i2,i3,i4)*NumberOP(i1,j1)",1.0); assert(success);
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoAux(i1,j2,i3,i4)*NumberOP(i2,j2)",+xi01); assert(success);
    //// right 
    success = exatn::initTensor("RhoAux",0.0); assert(success); // reset rho_aux
    success = exatn::contractTensors("RhoAux(i1,i2,i3,i4)+=RhoInRe(i1,i2,j3,i4)*NumberOP(j3,i3)",1.0); assert(success); 
    success = exatn::contractTensors("RhoOutIm(i1,i2,i3,i4)+=RhoAux(i1,i2,i3,j4)*NumberOP(j4,i4)",-xi01); assert(success);

     std::cout<< "("<< MPI_Wtime() - Timecurr << " sec)";
    Timecurr = MPI_Wtime();
  } // end of iexec loop
 
  // Get time
  double TimeExaStop = MPI_Wtime();

 // Synchronize for some reason (when? should this be in the loop above?) This is slow...
 std::cout<< "\nSyncing ExaTN... ";
 success = exatn::sync(); assert(success);
 double TimeExaSync = MPI_Wtime() - Timecurr;
 std::cout<< "Done ("<< TimeExaSync << " sec)." << std::endl;


  //std::cout<<" ######## ExaTN result ########" << std::endl;
  //printTensor("RhoOutRe");
  //printTensor("RhoOutIm");


  
  /* ----------------------------------------*/
  /* --- Compare Petsc and ExaTn results --- */
  /* ----------------------------------------*/

  auto local_copy_Re = exatn::getLocalTensor("RhoOutRe"); assert(local_copy_Re); //type = talsh::Tensor
  auto local_copy_Im = exatn::getLocalTensor("RhoOutIm"); assert(local_copy_Im); //type = talsh::Tensor
  auto tensor_view_Re = local_copy_Re->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //full tensor view
  auto tensor_view_Im = local_copy_Im->getSliceView<exatn::TensorDataType<TENS_ELEM_TYPE>::value>(); //full tensor view

  double err_re = 0.0;
  double err_im = 0.0;
  double norm_petsc = 0.0;
  for (int i0=0; i0<nlevels[0]; i0++){
    for (int i1=0; i1<nlevels[1]; i1++){
      for (int i0p=0; i0p<nlevels[0]; i0p++){
        for (int i1p=0; i1p<nlevels[1]; i1p++){
            // ExaTN values
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
            VecGetValues(rho_out_petsc, 1, &vecID_re, &valRe_Petsc);
            VecGetValues(rho_out_petsc, 1, &vecID_im, &valIm_Petsc);
            norm_petsc += pow(valRe_Petsc, 2.0);
            norm_petsc += pow(valIm_Petsc, 2.0);

            // error
            err_re += pow(valRe_exaTN - valRe_Petsc, 2.0);
            err_im += pow(valIm_exaTN - valIm_Petsc, 2.0);
        }
      }
    }
  }
  norm_petsc = sqrt(norm_petsc);
  err_re = sqrt(err_re)/norm_petsc/nexec;
  err_im = sqrt(err_im)/norm_petsc/nexec;
  std::cout<< std::endl<<std::endl << "Rel. avg. error (Petsc Vs ExaTN) = " << err_re << " + i " << err_im << std::endl << std::endl;

  local_copy_Re.reset();
  local_copy_Im.reset();


  /* Print timing */
  double TimeExa   = (TimeExaStop - TimeExaStart)/nexec;
  double TimePetsc = (TimePetscStop - TimePetscStart)/nexec ;
  std::cout<< "average time ExaTN ("<< nexec << " exec) : " << TimeExa   << " sec" << "  (+ Sync " << TimeExaSync << " sec) "<< std::endl;
  std::cout<< "average time PetsC ("<< nexec << " exec) : " << TimePetsc << " sec" << std::endl;
  std::cout<<std::endl;

 }

 exatn::finalize();
#ifdef MPI_ENABLED
 mpi_error = MPI_Finalize(); assert(mpi_error == MPI_SUCCESS);
#endif
 return 0;
}

