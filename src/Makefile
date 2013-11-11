main: common.o kernels.o main.o DRand.o
	nvcc -g -G -l cudart -l cublas common.o kernels.o DRand.o main.o -o main

common.o: 
	nvcc -g -G -I. -c common.cu

kernels.o:
	nvcc -g -G -I. -c kernels.cu

DRand.o:
	nvcc -g -G -I. --compiler-options "-std=c++0x" -c DRand.cc

main.o: 
	nvcc -g -G -I. -c main.cu


clean:
	rm *.o main
