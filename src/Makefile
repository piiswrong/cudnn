main: common.o kernels.o main.o DRand.o
	nvcc -g -G -l cudart -l cublas common.o kernels.o DRand.o main.o -o main
	rm main.o

common.o: common.cuh common.cu 
	nvcc -g -G -I. -c common.cu

kernels.o: kernels.cuh kernels.cu
	nvcc -g -G -I. -c kernels.cu

DRand.o: DRand.h DRand.cc
	nvcc -g -G -I. --compiler-options "-std=c++0x" -c DRand.cc

main.o: main.cu
	nvcc -g -G -I. -c main.cu


clean:
	rm *.o main
