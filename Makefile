# Set location of PETSC
#PETSC_DIR=/usr/local/Cellar/petsc/3.20.2
#PETSC_ARCH=arch-linux-c-debug

# Optional: Link to SLEPC
WITH_SLEPC = false
#SLEPC_DIR=/path/to/slepc-<version>

# Choose to run sanity tests
SANITY_CHECK = false

# Choose to use Ensmallen optimizer
# g++ example.cpp -o example -O2 -larmadillo -std=c++11 -I/Users/guenther5/Software/ensmallen-install/include 
WITH_ENSMALLEN=true
ENSM_DIR=/Users/guenther5/Software/ensmallen-install


#######################################################
# Typically no need to change anything below
#######################################################

# Add optional Slepc
ifeq ($(WITH_SLEPC), true)
CXX_OPT = -DWITH_SLEPC
LDFLAGS_OPT = -L${SLEPC_DIR}/lib -L${SLEPC_DIR}/${PETSC_ARCH}/lib -lslepc 
INC_OPT = -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${SLEPC_DIR}/include
endif

# Add optional Ensmallen
ifeq ($(WITH_ENSMALLEN), true)
CXX_OPT += -DWITH_ENSMALLEN
LDFLAGS_OPT += -L${ENSM_DIR}/lib -larmadillo
INC_OPT += -I${ENSM_DIR}/include
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
LDFLAGS =  -lm  -L${PETSC_DIR}/${PETSC_ARCH}/lib -lblas -llapack ${LDFLAGS_OPT} -lpetsc -std=c++14

# Set compiler and flags 
CXX=mpicxx
CXXFLAGS= -O3 -std=c++14 $(CXX_OPT)


# Rule for linking main executable 'quandary'
quandary: $(OBJ_FILES)
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
	rm -f  quandary

# use 'make clean-regtest' to remove tests/results
clean-regtest:
	rm -rf tests/results/*
	rm -rf tests/*/data_out

# include the dependency files
-include $(OBJ_FILES:.o=.d)
