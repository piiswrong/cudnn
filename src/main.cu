#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>

int main(int argc, char **argv) {
#ifdef USE_MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
#endif
    cublasHandle_t handle = 0; 
    CUDA_CALL(cublasCreate(&handle));

    int num_layers = 2;
    int layer_dims[] = {28*28, 1024, 10};
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    //_bp_hyper_params.batch_size = 10;
    _bp_hyper_params.check_interval = 10000;
    _bp_hyper_params.learning_rate = 1.0;
    _bp_hyper_params.idrop_out = false;
    _bp_hyper_params.hdrop_out = false;
    _bp_hyper_params.momentum = 0.0;
    _bp_hyper_params.max_momentum = 0.0;
#ifdef ADMM
    _bp_hyper_params.decay_rate = 0.01;
#endif

    //_bp_hyper_params.sparseInit = true;
    DNeuron<float> **neuron = new DNeuron<float>*[num_layers];
    neuron[0] = new DReLUNeuron<float>(handle);
    neuron[1] = new DSoftmaxNeuron<float>(_bp_hyper_params.batch_size, handle);
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neuron, _pt_hyper_params, _bp_hyper_params, handle);
#ifdef ADMM
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size, mpi_world_rank, _bp_hyper_params.batch_size, dnn->handle());
    dnn->admmFineTune(data, 50);
#else
    DMnistData<float> *data = new DMnistData<float>("../data", DData<float>::Train, 50000, false, dnn->handle());
    //DData<float> *data = new DDummyData<float>(10,  handle);
    dnn->fineTune(data, 10);
#endif

    DMnistData<float> *test_data;// = new DMnistData<float>("../data", DData<float>::Test, 10000, false, dnn->handle());
    test_data = new DMnistData<float>("../data", DData<float>::Test, 10000, true, dnn->handle());
    printf("Testing Error:%f\n", dnn->test(test_data));

    CUDA_CALL(cudaDeviceReset());
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
