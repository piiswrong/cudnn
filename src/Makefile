#FLAGS = -O3 -DNDEBUG --use_fast_math 
FLAGS = -g -G
#FLAGS = -O3 -DNDEBUG -DDISABLE_GPU -DWITH_GMTK
#FLAGS =  -g -DDISABLE_GPU #-DADMM
#SUFIX = cpp 
SUFIX = cu
#CC = gcc -pthread -std=c++0x
CC = nvcc --compiler-options="-pthread" 
#FLAGS += -DNVML
LIBS = -lcudart -lcublas 
#LIBS += -lnvidia-ml
LIBS += -L/opt/intel/mkl/lib/intel64 
LIBS += -L/opt/intel/lib/intel64 
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/mkl/lib/intel64
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/lib/intel64
LIBS += -L/projects/grail/local/pkg/intel/mkl/lib/em64t
#LIBS += -L/usr/lib
LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
LIBS += -lstdc++
INCLUDE = -I.
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
