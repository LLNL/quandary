# Set location of PETSC
#PETSC_DIR = /usr/tce/packages/petsc/petsc-3.8.3-mvapich2-2.2-gcc-4.8-redhat

# Set Braid location 
BRAID_DIR = ${HOME}/Numerics/xbraid

# Some braid vars
BRAID_INC_DIR = $(BRAID_DIR)/braid
BRAID_LIB_FILE = $(BRAID_DIR)/braid/libbraid.a


# Include some petsc libs, these might change depending on the example you run
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
#
# Set direction of source and header files, and build direction
SRC_DIR   = src
INC_DIR   = include
BUILD_DIR = build

# list all source and object files
SRC_FILES  = $(wildcard $(SRC_DIR)/*.cpp)
SRC_FILES += $(wildcard $(SRC_DIR)/*/*.cpp)
OBJ_FILES  = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

# set include directory
INC = -I$(INC_DIR) -I$(BRAID_INC_DIR) -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include

# Set Library paths and flags
LDPATH  = ${PETSC_DIR}/${PETSC_ARCH}/lib
LDFLAGS = -lpetsc -lm ${BRAID_LIB_FILE}

CC=mpicxx

# Rule for linking main
main: $(OBJ_FILES)
	$(CC) -o $@ $(OBJ_FILES) $(LDFLAGS) -L$(LDPATH) -g

# Rule for building all src files
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) -c $< -o $@ $(INC) -g 
	@$(CC) -MM $< -MP -MT $@ -MF $(@:.o=.d) $(INC) -g 


.PHONY: all cleanup

# use 'make cleanup' to remove object files and executable
cleanup:
	rm -fr $(BUILD_DIR) 
	rm -f  main 


# include the dependency files
-include $(OBJ_FILES:.o=.d)
