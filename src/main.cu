#include <common.cuh>
#include <DHyperParams.h>
#include <DDataSpec.h>
#include <DOption.h>
#include <DNN.cuh>
#include <DData.cuh>
#include <DKmeans.cuh>
#include <time.h>
#include <iostream>
#include <fstream>


#ifdef NVML
#include <nvml_old.h>
#endif
#include <sstream>

template<class T>
void fineTuneWithCheckpoint(DNN<T> *dnn, DData<T> *data, int ntotal, int ninterval, std::string path, int resuming) {
    for (int i = 0; i < ntotal; i += ninterval) {
        if (i + ninterval <= ntotal) 
            dnn->fineTune(data, i+resuming, i+resuming+ninterval);
        else 
            dnn->fineTune(data, i+resuming, i+resuming+ntotal);
        std::stringstream ss;
        ss << i+resuming;
        FILE *fout = fopen((path+"_"+ss.str()+".param").c_str(), "w");
        dnn->save(fout);
        fclose(fout);
    }
}

int main(int argc, char **argv) {
    std::string path = "/projects/grail/jxie/cudnn/log/";
    srand(time(0));
    //feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#ifdef USE_MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
#endif

    DOption opt;
    opt.num_layers = 2;
    opt.hidden_dim = 1023;
    opt.input_dim = 10, opt.output_dim = 255;
    opt.neuron = "ReLU";
    opt.pt_epochs = 0.0;
    opt.bp_epochs = 20;
    opt.pt_hyper_params.idrop_out = false;
    opt.pt_hyper_params.idrop_rate = 0.2;
    opt.pt_hyper_params.hdrop_out = false;
    opt.pt_hyper_params.weight_decay = false;
    opt.pt_hyper_params.decay_rate = 0.00;
    opt.pt_hyper_params.momentum = 0.90;
    opt.pt_hyper_params.learning_rate = 0.01;

    opt.bp_hyper_params.check_interval = 500;
    opt.bp_hyper_params.learning_rate = 0.1;
    opt.bp_hyper_params.learning_rate_decay = 0.00000;
    opt.bp_hyper_params.idrop_out = false;
    opt.bp_hyper_params.idrop_rate = 0.2;
    opt.bp_hyper_params.hdrop_out = false;
    opt.bp_hyper_params.hdrop_rate= 0.2;
    opt.bp_hyper_params.momentum = 0.5;
    opt.bp_hyper_params.max_momentum = 0.90;
    opt.bp_hyper_params.step_momentum = 0.04;
    opt.bp_hyper_params.weight_decay = false;
    opt.bp_hyper_params.decay_rate = 0.01;

    opt.bp_hyper_params.batch_size = 128;

#ifdef ADMM
    opt.bp_hyper_params.decay_rate = 0.001;
#endif

    if (!opt.parse(argc, argv)) {
        exit(-1);
    }
    if (opt.log_path != "") flog = fopen(opt.log_path.c_str(), "w");

#ifndef DISABLE_GPU
#ifdef NVML
    if (opt.devId == -1) {
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
                    opt.devId = i;
                    break;
                }
            }
        }
    }

    if (opt.devId != -1)
    {
        printf("Selecting device %d\n", opt.devId);
        CUDA_CALL(cudaSetDevice(opt.devId));
    }else {
        printf("Can not find idle device\n");
        exit(-1);
    }
#else
    if (opt.devId == -1) opt.devId = 0;
#endif
#endif


    cublasHandle_t handle = 0; 
    CUBLAS_CALL(cublasCreate(&handle));


    int *layer_dims = new int[opt.num_layers+1];
    layer_dims[0] = opt.input_dim;
    layer_dims[opt.num_layers] = opt.output_dim;
    for (int i = 1; i < opt.num_layers; i++) layer_dims[i] = opt.hidden_dim;

    DNeuron<float> **neurons = new DNeuron<float>*[opt.num_layers];
    for (int i = 0; i < opt.num_layers-1; i++) {
        if (opt.neuron == "Logistic") {
            neurons[i] = new DLogisticNeuron<float>(handle);
        }else if (opt.neuron == "Oddroot") {
            neurons[i] = new DOddrootNeuron<float>(handle);
        }else if (opt.neuron == "ReLU") {
            neurons[i] = new DReLUNeuron<float>(handle);
        }else if (opt.neuron == "Linear") {
            neurons[i] = new DNeuron<float>(handle);
        }else if (opt.neuron == "Tanh") {
            neurons[i] = new DTanhNeuron<float>(handle);
        }else if (opt.neuron == "Cutoff") {
            neurons[i] = new DCutoffNeuron<float>(handle);
        }else {
            printf("ERROR: \"%s\" is not a supported neuron type\n", opt.neuron.c_str());
            exit(-1);
        }
    }
    neurons[opt.num_layers-1] = new DSoftmaxNeuron<float>(opt.bp_hyper_params.batch_size, handle);
#ifdef ADMM
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size, mpi_world_rank, opt.bp_hyper_params.batch_size, dnn->handle());
    data->set_devId(opt.devId);
    dnn->admmFineTune(data, 500);
#elif defined(DOWN_POUR_SGD)
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size - sgd_num_param_server, mpi_world_rank - sgd_num_param_server, opt.bp_hyper_params.batch_size, dnn->handle());
    data->set_devId(opt.devId);
    dnn->fineTune(data, 500);

#else
    //DMnistData<float> *data = new DMnistData<float>("../data/", DData<float>::Train, 50000, false, dnn->handle());
    //DData<float> *data = new DDummyData<float>(input_dim, 1, handle);
    //DTimitData<float> *data = new DTimitData<float>("/scratch/jxie/", 10000, false, handle);
    //DData<float> *data = new DPatchData<float>("/projects/grail/jxie/paris/", input_dim, 10000, false, handle);
    DData<float> *data;
    if (opt.count("data")) {
        data = new DGeneralData<float,float,int>(opt.data_spec, opt.bp_hyper_params.batch_size*100, false, false, handle);
    }else {
        data = new DPatchData<float>("../data/", opt.input_dim, 2000, false, handle);
        //data = new DRankData<float>("../data/", input_dim, 64, false, handle);
    }
#ifndef DISABLE_GPU
    data->set_devId(opt.devId);
#endif

    //neurons[num_layers-1] = new DTanhNeuron<float>(handle);
    //neurons[num_layers-1] = new DGMMNeuron<float>(&opt.bp_hyper_params, 256, output_dim, 0.1, handle);
    //DvMFNeuron<float> *last_neuron = new DvMFNeuron<float>(&opt.bp_hyper_params, 32, output_dim, 0.2, handle);
    DClusterNeuron2<float> *last_neuron = new DClusterNeuron2<float>(&opt.bp_hyper_params, 255, opt.output_dim, 0.8, 10.0, handle);
    neurons[opt.num_layers-1] =  last_neuron;
    //last_neuron->init(data);
    //neurons[num_layers-1] = last_neuron;
    
    DNN<float> *dnn = new DNN<float>(opt.num_layers, layer_dims, neurons, &opt.pt_hyper_params, &opt.bp_hyper_params, handle);

    if (opt.grad_check) {
        return !dnn->createGradCheck(data);
    }

    DKmeans<float> *kmeans = new DKmeans<float>(dnn, data, opt.bp_hyper_params.batch_size, last_neuron->centers(), last_neuron->mask(), last_neuron->min_dist(), handle);

    //kmeans->cluster();

    
    if (opt.resuming == -1 && opt.pt_epochs > 0) dnn->pretrain(data, opt.pt_epochs);
    if (opt.resuming != -1) {
        printf("Resuming from %d-th epoch.\n", opt.resuming);
        //std::stringstream ss;
        //ss << opt.resuming - 10;
        FILE *fin = fopen(opt.input_path.c_str(), "r");
        if (fin == 0) {
            printf("Error loading: cannot find file %s!\n", opt.input_path.c_str());
            exit(-1);
        }
        dnn->layers()[0]->weight()->samplePrint();
        dnn->load(fin);
        dnn->layers()[0]->weight()->samplePrint();
        fclose(fin);
    }else 
        opt.resuming = 0;
    if (opt.master_file != "")
        fineTuneWithCheckpoint(dnn, data, opt.bp_epochs, 10, opt.master_file, opt.resuming);
    else {
        dnn->fineTune(data, opt.resuming, opt.resuming+opt.bp_epochs);
    }

#endif
    if (opt.output_path != "") {
        FILE *fout = fopen(opt.output_path.c_str(), "w");
        dnn->save(fout);
        fclose(fout);
    }
    if (opt.count("test")) {
        data->stop();
        data->set_testing(true);
        printf("Testing Loss: %f\n", dnn->test(data, opt.pred_path));
    }
    //DMnistData<float> *test_data;// = new DMnistData<float>("../data", DData<float>::Test, 10000, false, dnn->handle());
    //test_data->set_devId(opt.devId);
    //test_data = new DMnistData<float>("../data", DData<float>::Test, 10000, true, dnn->handle());
    //printf("Testing Error:%f\n", dnn->test(test_data));

    CUDA_CALL(cudaDeviceReset());
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
