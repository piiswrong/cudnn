#FLAGS = -O3 -DNDEBUG --use_fast_math 
FLAGS = -O3 -DNDEBUG -DDISABLE_GPU
#C11 = --compiler-options "-std=c++0x"
C11 = -std=c++0x
#CC = nvcc
CC = gcc -x c++ 
LIBS = #-lcudart -lcublas
LIBS += -L/usr/nikola/pkgs/intel/.2011.2.137/mkl/lib
INCLUDE = -I.
INCLUDE += -I/usr/nikola/pkgs/intel/.2011.2.137/mkl/include

main: common.o kernels.o main.o DRand.o
	$(CC) $(FLAGS) $(LIBS) common.o kernels.o DRand.o main.o -o main
	rm main.o

common.o: common.cuh common.cu 
	$(CC) $(FLAGS) $(INCLUDE) -c common.cu

kernels.o: kernels.cuh kernels.cu
	$(CC) $(FLAGS) $(INCLUDE) -c kernels.cu

DRand.o: DRand.h DRand.cc
	$(CC) $(FLAGS) $(INCLUDE) $(C11) -c DRand.cc

main.o: main.cu
	$(CC) $(FLAGS) $(INCLUDE) -c main.cu


clean:
	rm *.o main
