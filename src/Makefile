main: common.o kernels.o main.o DRand.o
	nvcc -l cudart -l cublas common.o kernels.o DRand.o main.o -o main

common.o: 
	nvcc -I. -c common.cu

kernels.o:
	nvcc -I. -c kernels.cu

DRand.o:
	nvcc -I. --compiler-options "-std=c++0x" -c DRand.cc

main.o: 
	nvcc -I. -c main.cu
