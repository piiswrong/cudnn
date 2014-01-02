#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>

#include <nvml_old.h>
#include <sstream>

template<class T>
void fineTuneWithCheckpoint(DNN<T> *dnn, DData<T> *data, int ntotal, int ninterval, std::string path) {
    for (int i = 0; i < ntotal; i += ninterval) {
        if (i + ninterval <= ntotal) 
            dnn->fineTune(data, ninterval);
        else 
            dnn->fineTune(data, ntotal - i);
        std::stringstream ss;
        ss << i;
        FILE *fout = fopen((path+"_"+ss.str()+".param").c_str(), "w");
        dnn->save(fout);
        fclose(fout);
    }
}

int main(int argc, char **argv) {
    std::string path = "/projects/grail/jxie/cudnn/log/";
//    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#ifdef USE_MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
#endif

    nvmlReturn_t ret;
    unsigned int deviceCount;

    if ((ret = nvmlInit()) != NVML_SUCCESS)
    {
        printf("Could not init NVML: %s\n", nvmlErrorString(ret));
        return 1;
    }

    if ((ret = nvmlDeviceGetCount(&deviceCount)) != NVML_SUCCESS)
    {
        printf("Could not get device count: %s\n", nvmlErrorString(ret));
        nvmlShutdown();
        return 1;
    }
    printf("Device count: %d\n", deviceCount);

    int devId = -1;
    for (int i = 0; i < deviceCount; i++) {
        nvmlDevice_t device;
        if ((ret = nvmlDeviceGetHandleByIndex(i, &device)) != NVML_SUCCESS)
        {
            printf("Skip %d, can not get index\n", i);
            continue;
        }
        nvmlUtilization_t util;
        if ((ret = nvmlDeviceGetUtilizationRates(device, &util)) != NVML_SUCCESS)
        {
            printf("Can not get util rate on %d\n", i);
        }else {
            if (util.gpu < 5) 
            {
                devId = i;
                break;
            }
        }
    }

    if (devId != -1)
    {
        printf("Selecting device %d\n", devId);
        CUDA_CALL(cudaSetDevice(devId));
    }else {
        printf("Can not find idle device\n");
        exit(-1);
    }
                                        


    FILE *param_out = NULL;
    FILE *fin = NULL;
    if (argc >= 2) {
        flog = fopen((path+argv[1]+".log").c_str(), "w");
        fin = fopen((path+argv[1]+".hyper").c_str(), "r");
        printf("%s\n", (path+argv[1]+".hyper").c_str());
        if (fin == NULL) exit(-1);
    }

    cublasHandle_t handle = 0; 
    CUBLAS_CALL(cublasCreate(&handle));

    int num_layers = 20;
    int hidden_dim = 647;
    char unit[255];
    strcpy(unit, "Logistic");
    double pt_epochs = 0.2;
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    //_bp_hyper_params.batch_size = 10;
    _pt_hyper_params.idrop_out = true;
    _pt_hyper_params.idrop_rate = 0.5;
    _pt_hyper_params.learning_rate = 0.0001;

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

    int bp_epochs = 200;
    if (fin != NULL) {
        READ_PARAM(num_layers);
        READ_PARAM(hidden_dim);
        fscanf(fin, "neuron=%s\n", unit);
        READ_PARAM(pt_epochs);
        READ_PARAM(bp_epochs);
        _pt_hyper_params.load(fin);
        _bp_hyper_params.load(fin);
    }


    int input_dim = 351, output_dim = 150;
    int *layer_dims = new int[num_layers+1];
    layer_dims[0] = input_dim;
    layer_dims[num_layers] = output_dim;
    for (int i = 1; i < num_layers; i++) layer_dims[i] = hidden_dim;

    DNeuron<float> **neuron = new DNeuron<float>*[num_layers];
    for (int i = 0; i < num_layers-1; i++) {
        std::string str_unit(unit);
        if (str_unit == "Logistic") {
            neuron[i] = new DLogisticNeuron<float>(handle);
        }else if (str_unit == "Oddroot") {
            neuron[i] = new DOddrootNeuron<float>(handle);
        }else if (str_unit == "ReLU") {
            neuron[i] = new DReLUNeuron<float>(handle);
        }else {
            printf("ERROR: \"%s\" is not a supported neuron type\n", unit);
            exit(-1);
        }
    }
    neuron[num_layers-1] = new DSoftmaxNeuron<float>(_bp_hyper_params.batch_size, handle);
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neuron, _pt_hyper_params, _bp_hyper_params, handle);
#ifdef ADMM
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size, mpi_world_rank, _bp_hyper_params.batch_size, dnn->handle());
    data->set_devId(devId);
    dnn->admmFineTune(data, 500);
#elif defined(DOWN_POUR_SGD)
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size - sgd_num_param_server, mpi_world_rank - sgd_num_param_server, _bp_hyper_params.batch_size, dnn->handle());
    data->set_devId(devId);
    dnn->fineTune(data, 500);

#else
    //DMnistData<float> *data = new DMnistData<float>("../data", DData<float>::Train, 50000, false, dnn->handle());
    //DData<float> *data = new DDummyData<float>(10,  handle);
    DTimitData<float> *data = new DTimitData<float>("/scratch/jxie/", 10000, false, dnn->handle());
    data->set_devId(devId);
    if (pt_epochs > 0) dnn->pretrain(data, pt_epochs);
    //dnn->fineTune(data, bp_epochs);
    fineTuneWithCheckpoint(dnn, data, bp_epochs, 10, path+argv[1]);
#endif
    if (param_out != NULL) {
        dnn->save(param_out);
    }
    DMnistData<float> *test_data;// = new DMnistData<float>("../data", DData<float>::Test, 10000, false, dnn->handle());
    //test_data->set_devId(devId);
    //test_data = new DMnistData<float>("../data", DData<float>::Test, 10000, true, dnn->handle());
    //printf("Testing Error:%f\n", dnn->test(test_data));

    CUDA_CALL(cudaDeviceReset());
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
