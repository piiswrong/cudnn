#FLAGS = -O3 -DNDEBUG --use_fast_math 
FLAGS = -O3 -DNDEBUG -DDISABLE_GPU -DDOWN_POUR_SGD#-DADMM
#FLAGS = -g -DDISABLE_GPU -DADMM
SUFIX = cpp
CC = mpicxx 
LIBS = #-lcudart -lcublas
LIBS += -L/opt/intel/mkl/lib/intel64 
LIBS += -L/opt/intel/lib/intel64 
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/mkl/lib/intel64
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/lib/intel64
#LIBS += -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -ldl
LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
LIBS += -lstdc++
INCLUDE = -I.
INCLUDE += -I/opt/intel/mkl/include
INCLUDE += -I/usr/nikola/pkgs/intel/.2011.2.137/mkl/include
INCLUDE += -I/usr/include/mpi
INCLUDE += -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi 

main: common.o kernels.o main.o 
	$(CC) $(FLAGS) $(LIBS) common.o kernels.o main.o -o main
	rm main.o

common.o: common.cuh common.$(SUFIX)
	$(CC) $(FLAGS) $(INCLUDE) -c common.$(SUFIX)

kernels.o: kernels.cuh kernels.$(SUFIX)
	$(CC) $(FLAGS) $(INCLUDE) -c kernels.$(SUFIX)

main.o: main.$(SUFIX)
	$(CC) $(FLAGS) $(INCLUDE) -c main.$(SUFIX)

clean:
	rm *.o main
