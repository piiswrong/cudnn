#FLAGS = -O3 -DNDEBUG --use_fast_math 
#FLAGS = -O3 -DNDEBUG -DDISABLE_GPU 
FLAGS = -g -DDISABLE_GPU
C11 = --compiler-options "-std=c++0x"
#C11 = -std=c++0x
CC = nvcc
LIBS = #-lcudart -lcublas
LIBS += -L/opt/intel/mkl/lib/intel64 
LIBS += -L/opt/intel/lib/intel64 
LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
INCLUDE = -I.
INCLUDE += -I/opt/intel/mkl/include

main: common.o kernels.o main.o DRand.o
	nvcc $(FLAGS) $(LIBS) common.o kernels.o DRand.o main.o -o main
	rm main.o

common.o: common.cuh common.cpp 
	$(CC) $(FLAGS) $(INCLUDE) -c common.cpp

kernels.o: kernels.cuh kernels.cpp
	$(CC) $(FLAGS) $(INCLUDE) -c kernels.cpp

DRand.o: DRand.h DRand.cc
	$(CC) $(FLAGS) $(INCLUDE) $(C11) -c DRand.cc

main.o: main.cpp
	$(CC) $(FLAGS) $(INCLUDE) -c main.cpp


clean:
	rm *.o main
