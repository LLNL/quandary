# Set location of PETSC
PETSC_DIR = /usr/tce/packages/petsc/petsc-3.8.3-mvapich2-2.2-gcc-4.8-redhat

# Include some petsc libs, these might change depending on the example you run
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# location of petsc library
LDPATH  = ${PETSC_DIR}/${PETSC_ARCH}/lib

# Set Braid location 
BRAID_DIR = ${HOME}/Numerics/xbraid

# Set flags
LDFLAGS = -lpetsc -lm ${BRAID_DIR}/braid/libbraid.a
CPPFLAGS= -I${BRAID_DIR}/braid

.PHONY: all cleanup

# Put all files that are to be compiled here (no ending)
ALL_PROGS = ex-01 ex-02 base dttest matrix_free two_oscillators

# Type 'make all' to compile and link all examples in $ALL_PROGs
all: $(ALL_PROGS)

# Type'make ex-01' to compile and link only ex-01.c
ex-01: ex-01.o
	$(CC) -L${LDPATH} -o $@ $^ $(LDFLAGS)

# Type'make ex-01' to compile and link only ex-01.c
ex-02: ex-02.o
		$(CC) -L${LDPATH} -o $@ $^ $(LDFLAGS)

# Type'make ex-02' to compile and link only ex-02.c
base: base.o
	$(CC) -L${LDPATH} -o $@ $^ $(LDFLAGS)

# Type'make ex-02' to compile and link only ex-02.c
dttest: dttest.o
	$(CC) -L${LDPATH} -o $@ $^ $(LDFLAGS)

matrix_free: matrix_free.o
	$(CC) -L${LDPATH} -o $@ $^ $(LDFLAGS)

two_oscillators: two_oscillators.o
	$(CC) $(CPPFLAGS) -L${LDPATH} -o $@ $^ $(LDFLAGS)


# use 'make cleanup' to remove object files and executable
cleanup:
	rm -f $(ALL_PROGS)
	rm -f *.o
