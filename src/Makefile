flags = -O3  --use_fast_math -DNDEBUG
main: common.o kernels.o main.o DRand.o
	nvcc $(flags) -l cudart -l cublas common.o kernels.o DRand.o main.o -o main
	rm main.o

common.o: common.cuh common.cu 
	nvcc $(flags) -I. -c common.cu

kernels.o: kernels.cuh kernels.cu
	nvcc $(flags) -I. -c kernels.cu

DRand.o: DRand.h DRand.cc
	nvcc $(flags) -I. --compiler-options "-std=c++0x" -c DRand.cc

main.o: main.cu
	nvcc $(flags) -I. -c main.cu


clean:
	rm *.o main
