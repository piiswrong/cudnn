#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>

int main(int argc, char **argv) {
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#ifdef USE_MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
#endif
    FILE *param_out = NULL;
    if (argc >= 2) {
        std::string path = "../log/";
        path = path + argv[1];
        flog = fopen((path+".log").c_str(), "w");
        param_out = fopen((path+".param").c_str(), "w");
    }

    cublasHandle_t handle = 0; 
    CUBLAS_CALL(cublasCreate(&handle));

    int num_layers = 3;
    int layer_dims[] = {351, 2047, 2047, 150};
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    //_bp_hyper_params.batch_size = 10;
    _bp_hyper_params.check_interval = 10000;
    _bp_hyper_params.learning_rate = 0.1;
    _bp_hyper_params.idrop_out = false;
    _bp_hyper_params.hdrop_out = true;
    _bp_hyper_params.hdrop_rate= 0.2;
    _bp_hyper_params.momentum = 0.5;
    _bp_hyper_params.max_momentum = 0.90;
    _bp_hyper_params.step_momentum = 0.04;
    _bp_hyper_params.weight_decay = false;
#ifdef ADMM
    _bp_hyper_params.decay_rate = 0.001;
#endif

    //_bp_hyper_params.sparseInit = true;
    DNeuron<float> **neuron = new DNeuron<float>*[num_layers];
    neuron[0] = new DOddrootNeuron<float>(handle);
    neuron[1] = new DOddrootNeuron<float>(handle);
    neuron[2] = new DSoftmaxNeuron<float>(_bp_hyper_params.batch_size, handle);
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neuron, _pt_hyper_params, _bp_hyper_params, handle);
#ifdef ADMM
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size, mpi_world_rank, _bp_hyper_params.batch_size, dnn->handle());
    dnn->admmFineTune(data, 500);
#elif defined(DOWN_POUR_SGD)
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size - sgd_num_param_server, mpi_world_rank - sgd_num_param_server, _bp_hyper_params.batch_size, dnn->handle());
    dnn->fineTune(data, 500);

#else
    //DMnistData<float> *data = new DMnistData<float>("../data", DData<float>::Train, 50000, false, dnn->handle());
    //DData<float> *data = new DDummyData<float>(10,  handle);
    DTimitData<float> *data = new DTimitData<float>("../data", 128*100, false, dnn->handle());
    dnn->fineTune(data, 200);
#endif
    if (param_out != NULL) {
        dnn->save(param_out);
    }
    DMnistData<float> *test_data;// = new DMnistData<float>("../data", DData<float>::Test, 10000, false, dnn->handle());
    //test_data = new DMnistData<float>("../data", DData<float>::Test, 10000, true, dnn->handle());
    //printf("Testing Error:%f\n", dnn->test(test_data));

    CUDA_CALL(cudaDeviceReset());
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
