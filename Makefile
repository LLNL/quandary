# Set location of PETSC
#PETSC_DIR=/path/to/petsc
#PETSC_ARCH=arch-linux-c-debug

# Choose to link with XBraid and set the location
WITH_XBRAID = false
#BRAID_DIR = /path/to/xbraid_solveadjointwithxbraid

# Choose to link with the SLEPC library and set the location
WITH_SLEPC = false
#SLEPC_DIR=/path/to/slepc-3.13.3

# Choose to run sanity tests
SANITY_CHECK = false

#######################################################
# Typically no need to change anything below

# Add optional Slepc
ifeq ($(WITH_SLEPC), true)
CXX_OPT = -DWITH_SLEPC
LDFLAGS_OPT = -L${SLEPC_DIR}/lib -L${SLEPC_DIR}/${PETSC_ARCH}/lib -lslepc 
INC_OPT = -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${SLEPC_DIR}/include
endif


# Add optional Braid include and library location
ifeq ($(WITH_XBRAID), true)
BRAID_INC_DIR  = $(BRAID_DIR)/braid
BRAID_LIB_FILE = $(BRAID_DIR)/braid/libbraid.a
CXX_OPT += -DWITH_BRAID
INC_OPT += -I${BRAID_INC_DIR}
LDFLAGS_OPT += ${BRAID_LIB_FILE}
endif

# Add sanity check to compiler option
ifeq ($(SANITY_CHECK), true)
CXX_OPT += -DSANITY_CHECK
endif

# Include some petsc libs, these might change depending on the example you run
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# Set direction of source and header files, and build direction
SRC_DIR   = src
INC_DIR   = include
BUILD_DIR = build

# list all source and object files
SRC_FILES  = $(wildcard $(SRC_DIR)/*.cpp)
SRC_FILES += $(wildcard $(SRC_DIR)/*/*.cpp)
OBJ_FILES  = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

# set include directory
INC = -I$(INC_DIR) -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include ${INC_OPT}

# Set Library paths and flags
LDPATH  = ${PETSC_DIR}/${PETSC_ARCH}/lib
LDFLAGS = -lpetsc -lm  -L${PETSC_DIR}/${PETSC_ARCH}/lib -lblas -llapack ${LDFLAGS_OPT}

# Set compiler and flags 
CXX=mpicxx
CXXFLAGS= -O3 -std=c++11 -lstdc++ $(CXX_OPT)


# Rule for linking main
main: $(OBJ_FILES)
	$(CXX) -o $@ $(OBJ_FILES) $(LDFLAGS) -L$(LDPATH)

# Rule for building all src files
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@ $(INC) 
	@$(CXX) -MM $< -MP -MT $@ -MF $(@:.o=.d) $(INC) 

# Rule for building and linking main-tensor:
main-tensor: 
	mpicxx -c -std=gnu++14 -fPIC  -DPATH_MAX=4096 -Wno-attributes -DNO_GPU -DEXATN_SERVICE src-tensor/main-tensor.cpp -o src-tensor/main-tensor.o -Iinclude  -I/home/guenther5/.exatn/include/exatn -I/home/guenther5/.exatn/include -I/home/guenther5/.exatn/include/cppmicroservices4
	mpicxx -o main-tensor src-tensor/main-tensor.o -rdynamic -Wl,-rpath,/home/guenther5/.exatn/lib -L /home/guenther5/.exatn/lib -lCppMicroServices -ltalsh -lexatn -lexatn-numerics -lexatn-runtime -lexatn-runtime-graph -lexatn-utils -ldl -lpthread /usr/lib64/mpich/lib/libmpicxx.so /usr/lib64/mpich/lib/libmpi.so /usr/lib64/libblas.so /usr/lib/gcc/x86_64-redhat-linux/10/libgomp.so /usr/lib64/libpthread.so /usr/lib/gcc/x86_64-redhat-linux/10/libgomp.so /usr/lib64/libpthread.so -lgfortran /usr/lib64/liblapack.so -L/home/guenther5/Software/petsc/linux-gnu-c-debug/lib



.PHONY: all cleanup clean-regtest

# use 'make cleanup' to remove object files and executable
cleanup:
	rm -fr $(BUILD_DIR) 
	rm -f  main 
	rm -f  src-tensor/*.o
	rm -f  main-tensor

# use 'make clean-regtest' to remove tests/results
clean-regtest:
	rm -rf tests/results/*
	rm -rf tests/*/data_out

# Rule for cleaning up tensor
clean-tensor:
	rm -f  src-tensor/*.o
	rm -f  main-tensor

# include the dependency files
-include $(OBJ_FILES:.o=.d)
