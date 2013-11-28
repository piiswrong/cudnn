#FLAGS = -O3 -DNDEBUG --use_fast_math 
FLAGS = -O3 -DNDEBUG -DDISABLE_GPU 
#FLAGS = -g -DDISABLE_GPU
SUFIX = cpp
C11 = --compiler-options "-std=c++0x"
#C11 = -std=c++0x
CC = nvcc
LIBS = -lcudart -lcublas
LIBS += -L/opt/intel/mkl/lib/intel64 
LIBS += -L/opt/intel/lib/intel64 
LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
INCLUDE = -I.
INCLUDE += -I/opt/intel/mkl/include

main: common.o kernels.o main.o DRand.o
	$(CC) $(FLAGS) $(LIBS) common.o kernels.o DRand.o main.o -o main
	rm main.o

common.o: common.cuh common.$(SUFIX)
	$(CC) $(FLAGS) $(INCLUDE) -c common.$(SUFIX)
kernels.o: kernels.cuh kernels.$(SUFIX)
	$(CC) $(FLAGS) $(INCLUDE) -c kernels.$(SUFIX)

DRand.o: DRand.h DRand.cc
	$(CC) $(FLAGS) $(INCLUDE) $(C11) -c DRand.cc

main.o: main.$(SUFIX)
	$(CC) $(FLAGS) $(INCLUDE) -c main.$(SUFIX)


clean:
	rm *.o main
