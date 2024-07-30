ROCM_DIR ?= /opt/rocm
RCCL_DIR ?= ../../build/release
MPI_DIR  ?= /opt/ompi
MPI_INC_DIR ?= /usr/include/x86_64-linux-gnu/mpi
MPI_LIB_DIR ?= /usr/lib/x86_64-linux-gnu

INCLUDES = -I$(MPI_INC_DIR) -I$(MPI_DIR)/include -I$(RCCL_DIR)/include
LDFLAGS  = -L$(MPI_LIB_DIR) -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpi -lrccl

main: allreduce_mtype.cpp
	$(ROCM_DIR)/bin/hipcc allreduce_mtype.cpp -O1 -g -o allreduce_mtype $(INCLUDES) $(LDFLAGS)

clean:
	rm -f ./allreduce_mtype
