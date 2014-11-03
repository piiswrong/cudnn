FLAGS = -O0 -DNDEBUG --use_fast_math 
#FLAGS = -g -G
FLAGS += -DBOOST_NOINLINE='__attribute__ ((noinline))'
#FLAGS = -O3 -DNDEBUG -DDISABLE_GPU -DWITH_GMTK
#FLAGS =  -g -DDISABLE_GPU #-DADMM
#SUFIX = cpp 
#CC = gcc -pthread -std=c++0x
CC = nvcc --compiler-options="-pthread"
#FLAGS += -DNVML
LIBS += -lnvidia-ml
LIBS += -lcudart -lcublas 
LIBS += -L/homes/jxie/lib
LIBS += -L/homes/grail/jxie/lib
LIBS += -L/opt/intel/mkl/lib/intel64 
LIBS += -L/opt/intel/lib/intel64 
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/mkl/lib/intel64
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/lib/intel64
LIBS += -L/projects/grail/local/pkg/intel/mkl/lib/em64t
LIBS += -L/usr/lib64/nvidia
LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
LIBS += -lstdc++
LIBS += -lboost_program_options -lboost_filesystem -lboost_system
INCLUDE = -I.
INCLUDE += -I/homes/jxie/include
INCLUDE += -I/homes/grail/jxie/include
INCLUDE += -I/opt/intel/mkl/include
INCLUDE += -I/usr/nikola/pkgs/intel/.2011.2.137/mkl/include
INCLUDE += -I/projects/grail/local/pkg/intel/mkl/include/
INCLUDE += -I/usr/include/mpi
INCLUDE += -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi 

#LIBS += -L../../tksrc -L../../miscSupport -L../../featureFileIO -L../../IEEEFloatingpoint -L../../deepMLP
#LIBS += -lIEEEsupport -lmiscSupport -lGMTK -lDMLP -lgmtkio
#INCLUDE += -I../../tksrc -I../../miscSupport -I../../featureFileIO -I../../IEEEFloatingpoint -I../../deepMLP -I../../mitk
#FLAGS += -DHAVE_CONFIG_H -DHAVE_HG_H

#gmtkGPUtrain: common.o kernels.o main.o
#	$(CC) $(FLAGS) $(INCLUDE) -c gmtkGPUtrain.cpp 
#	$(CC) $(FLAGS) $(LIBS) common.o kernels.o gmtkGPUtrain.o -o gmtkGPUtrain
#	rm gmtkGPUtrain.o
#

HEADERS = $(wildcard *.cuh)

main: common.o kernels.o main.o DDataSpec.o DHyperParams.o DOption.o
	$(CC) $(FLAGS) $(LIBS) common.o kernels.o main.o DDataSpec.o DHyperParams.o DOption.o -o main

common.o: common.cuh common.cu
	$(CC) $(FLAGS) $(INCLUDE) -c common.cu

kernels.o: kernels.cuh kernels.cu
	$(CC) $(FLAGS) $(INCLUDE) -c kernels.cu

main.o: main.cu $(HEADERS)
	$(CC) $(FLAGS) $(INCLUDE) -c main.cu

DHyperParams.o: DHyperParams.cc DHyperParams.h
	$(CC) $(FLAGS) $(INCLUDE) -c DHyperParams.cc

DDataSpec.o: DDataSpec.cc DDataSpec.h
	$(CC) $(FLAGS) $(INCLUDE) -c DDataSpec.cc

DOption.o: DOption.cc DOption.h
	$(CC) $(FLAGS) $(INCLUDE) -c DOption.cc


clean:
	rm -f *.o main 
