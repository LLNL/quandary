# Set location of PETSC
#PETSC_DIR=/path/to/petsc-<version>
#PETSC_ARCH=arch-linux-c-debug

# Optional: Link to SLEPC
WITH_SLEPC = false
#SLEPC_DIR=/path/to/slepc-<version>

# Optional: Enable the python interface
WITH_PYTHON = false
PYTHON_INCDIR = /usr/local/Caskroom/miniconda/base/envs/numpy-env/include/python3.9/  # location of Python.h. Try "python<version>-config --includes" to find it.
PYTHON_LIBDIR = /usr/local/Caskroom/miniconda/base/envs/numpy-env/lib/     # location of libpython<version>.so or libpython<version>.dylib (Mac). Try "python<version>-config --ldflags" to find it.
PYTHON_VERSION = 3.9   # Set the python version. This is not be needed if the libpython.so / libpython.dylib links to the correct version library libpython<version>.so
# Note: You'll have to set the environment variable LD_LIBRARY_PATH to include the PYTHON_LIBDIR path. For example: "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Caskroom/miniconda/base/envs/numpy-env/lib"

# Optional: If using the python interface, you can link with FitpackPP to enable spline-based transfer functions (optional).
WITH_FITPACK = false
FITPACK_DIR=${HOME}/Software/fitpackpp

# Choose to run sanity tests
SANITY_CHECK = false


#######################################################
# Typically no need to change anything below
#######################################################

# Add optional Slepc
ifeq ($(WITH_SLEPC), true)
CXX_OPT = -DWITH_SLEPC
LDFLAGS_OPT = -L${SLEPC_DIR}/lib -L${SLEPC_DIR}/${PETSC_ARCH}/lib -lslepc 
INC_OPT = -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${SLEPC_DIR}/include
endif


# Add optional Python interface
ifeq ($(WITH_PYTHON), true)
INC_OPT += -I${PYTHON_INCDIR}
LDFLAGS_OPT += -L${PYTHON_LIBDIR} -lpython${PYTHON_VERSION}
CXX_OPT += -DWITH_PYTHON
endif


# Add optional FITPACKPP for spline interpolation
ifeq ($(WITH_FITPACK), true)
INC_OPT += -I${FITPACK_DIR}/fitpackpp
LDFLAGS_OPT += -L${FITPACK_DIR}/build -lfitpackpp -lfitpack 
CXX_OPT += -DWITH_FITPACK
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
LDFLAGS =  -lm  -L${PETSC_DIR}/${PETSC_ARCH}/lib -lblas -llapack ${LDFLAGS_OPT} -lpetsc

# Set compiler and flags 
CXX=mpicxx
CXXFLAGS= -O3 -std=c++14 $(CXX_OPT)


# Rule for linking main
main: $(OBJ_FILES)
	$(CXX) -o $@ $(OBJ_FILES) $(LDFLAGS) -L$(LDPATH)

# Rule for building all src files
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@ $(INC) 
	@$(CXX) -MM $< -MP -MT $@ -MF $(@:.o=.d) $(INC) 


.PHONY: all cleanup clean-regtest

# use 'make cleanup' to remove object files and executable
cleanup:
	rm -fr $(BUILD_DIR) 
	rm -f  main 

# use 'make clean-regtest' to remove tests/results
clean-regtest:
	rm -rf tests/results/*
	rm -rf tests/*/data_out

# include the dependency files
-include $(OBJ_FILES:.o=.d)
