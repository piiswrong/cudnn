#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>
#include <tclap/CmdLine.h>

#ifdef NVML
#include <nvml_old.h>
#endif
#include <sstream>

template<class T>
void fineTuneWithCheckpoint(DNN<T> *dnn, DData<T> *data, int ntotal, int ninterval, std::string path, int resuming) {
    for (int i = 0; i < ntotal; i += ninterval) {
        if (i + ninterval <= ntotal) 
            dnn->fineTune(data, ninterval);
        else 
            dnn->fineTune(data, ntotal - i);
        std::stringstream ss;
        ss << i+resuming;
        FILE *fout = fopen((path+"_"+ss.str()+".param").c_str(), "w");
        dnn->save(fout);
        fclose(fout);
    }
}

int main(int argc, char **argv) {
    std::string path = "/projects/grail/jxie/cudnn/log/";
    //feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#ifdef USE_MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
#endif

    FILE *fin = NULL;
    std::string exp_name, log_path, hyper_path, input_path, output_path, pred_path;
    int resuming = -1;
    int devId = -1;
    bool grad_check = false;

    int num_layers = 2;
    int hidden_dim = 16;
    //int input_dim = 351, output_dim = 150;
    //int input_dim = 1568, output_dim = 256;
    //int input_dim = 28*28, output_dim = 10;
    int input_dim = 6, output_dim = 1;
    std::string neuron("Tanh");
    float pt_epochs = 0.0;
    int bp_epochs = 100000;
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    _pt_hyper_params.idrop_out = false;
    _pt_hyper_params.idrop_rate = 0.2;
    _pt_hyper_params.hdrop_out = false;
    _pt_hyper_params.weight_decay = false;
    _pt_hyper_params.decay_rate = 0.00;
    _pt_hyper_params.momentum = 0.90;
    _pt_hyper_params.learning_rate = 0.01;

    _bp_hyper_params.check_interval = 64;
    _bp_hyper_params.learning_rate = 0.5;
    _bp_hyper_params.learning_rate_decay = 1;
    _bp_hyper_params.idrop_out = false;
    _bp_hyper_params.idrop_rate = 0.2;
    _bp_hyper_params.hdrop_out = false;
    _bp_hyper_params.hdrop_rate= 0.5;
    _bp_hyper_params.momentum = 0.9;
    _bp_hyper_params.max_momentum = 0.90;
    _bp_hyper_params.step_momentum = 0.00;
    _bp_hyper_params.weight_decay = false;
    _bp_hyper_params.decay_rate = 0.01;

    _bp_hyper_params.batch_size = 64;

#ifdef ADMM
    _bp_hyper_params.decay_rate = 0.001;
#endif

    try{
        TCLAP::CmdLine cmd("", ' ', "0.1");
        TCLAP::UnlabeledValueArg<std::string> argExpName("expName", "Name of the experiment", false, "", "string", cmd);
        TCLAP::ValueArg<int> argResuming("r", "resuming", "Resume from n-th epoch", false, -1, "int", cmd); 
        TCLAP::ValueArg<int> argDevId("d", "dev-id", "Id of device to use", false, -1, "int", cmd); 
        TCLAP::SwitchArg argCheckGrad("c", "check-grad", "Check gradient", cmd, false);
        TCLAP::ValueArg<std::string> argLogPath("l", "log-path", "Path to log file", false, "", "string", cmd); 
        TCLAP::ValueArg<std::string> argHyperPath("h", "hyper-path", "Path to hyperparamter file", false, "", "string", cmd); 
        TCLAP::ValueArg<std::string> argInputPath("i", "input-path", "Path to input parameter file", false, "", "string", cmd); 
        TCLAP::ValueArg<std::string> argOutputPath("o", "output-path", "Path to output parameter file", false, "", "string", cmd); 
        TCLAP::ValueArg<std::string> argPredPath("t", "testing", "Path to prediction result output file", false, "", "string", cmd); 
        TCLAP::ValueArg<int> argVerbosity("v", "verbose", "log output verbosity level", false, VERBOSE_NORMAL, "int", cmd); 

        TCLAP::ValueArg<int> argNumLayers("", "numLayers", "number of layers", false, num_layers, "int", cmd);
        TCLAP::ValueArg<int> argHiddenDim("", "hiddenDim", "Dimension of hidden layers", false, hidden_dim, "int", cmd); 
        TCLAP::ValueArg<std::string> argNeuron("", "neuron", "Type of neuron to use", false, neuron, "string", cmd); 

        TCLAP::ValueArg<double> argBpLearningRate("", "bpLearningRate", "Initial learning rate", false, _bp_hyper_params.learning_rate, "double", cmd);
        TCLAP::ValueArg<double> argBpMomentum("", "bpMomentum", "Initial momentum", false, _bp_hyper_params.momentum, "double", cmd);

        cmd.parse(argc, argv);

        exp_name = argExpName.getValue();
        resuming = argResuming.getValue();
        devId = argDevId.getValue();
        grad_check = argCheckGrad.getValue();
        log_path = argLogPath.getValue();
        hyper_path = argHyperPath.getValue();
        input_path = argInputPath.getValue();
        output_path = argOutputPath.getValue();
        pred_path = argPredPath.getValue();
        log_verbosity = argVerbosity.getValue();

        if (exp_name != "") {
            if (log_path == "") log_path = path+exp_name+".log";
            if (hyper_path == "") hyper_path = path+exp_name+".hyper";
            if (input_path == "") input_path = path+exp_name+".param";
            if (output_path == "") output_path = path+exp_name+".param";
        }
        if (log_path != "") flog = fopen(log_path.c_str(), "w");
        if (hyper_path != "") {
            fin = fopen(hyper_path.c_str(), "r");
            if (fin == NULL) {
                printf("Cannot open configuration file %s\n", hyper_path.c_str());
                exit(-1);
            }
            printf("Using configuration file %s\n", hyper_path.c_str());
            READ_PARAM(num_layers);
            READ_PARAM(hidden_dim);
            char    tmp[255];
            fscanf(fin, "neuron=%s\n", tmp);
            neuron = tmp;
            READ_PARAM(pt_epochs);
            READ_PARAM(bp_epochs);
            _pt_hyper_params.load(fin);
            _bp_hyper_params.load(fin);
            fclose(fin);
        }

        num_layers = argNumLayers.getValue();
        hidden_dim = argHiddenDim.getValue();
        neuron = argNeuron.getValue();

    }catch(TCLAP::ArgException &e) {
        std::cout << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(-1);
    }

#ifndef DISABLE_GPU
#ifdef NVML
    if (devId == -1) {
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
                    devId = i;
                    break;
                }
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
#else
    devId = 0;
#endif
#endif


    cublasHandle_t handle = 0; 
    CUBLAS_CALL(cublasCreate(&handle));


    int *layer_dims = new int[num_layers+1];
    layer_dims[0] = input_dim;
    layer_dims[num_layers] = output_dim;
    for (int i = 1; i < num_layers; i++) layer_dims[i] = hidden_dim;

    DNeuron<float> **neurons = new DNeuron<float>*[num_layers];
    for (int i = 0; i < num_layers; i++) {
        if (neuron == "Logistic") {
            neurons[i] = new DLogisticNeuron<float>(handle);
        }else if (neuron == "Oddroot") {
            neurons[i] = new DOddrootNeuron<float>(handle);
        }else if (neuron == "ReLU") {
            neurons[i] = new DReLUNeuron<float>(handle);
        }else if (neuron == "Linear") {
            neurons[i] = new DNeuron<float>(handle);
        }else if (neuron == "Tanh") {
            neurons[i] = new DTanhNeuron<float>(handle);
        }else if (neuron == "Cutoff") {
            neurons[i] = new DCutoffNeuron<float>(handle);
        }else {
            printf("ERROR: \"%s\" is not a supported neuron type\n", neuron.c_str());
            exit(-1);
        }
    }
#ifdef ADMM
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size, mpi_world_rank, _bp_hyper_params.batch_size, dnn->handle());
    data->set_devId(devId);
    dnn->admmFineTune(data, 500);
#elif defined(DOWN_POUR_SGD)
    DParallelMnistData<float> *data = new DParallelMnistData<float>("../data", mpi_world_size - sgd_num_param_server, mpi_world_rank - sgd_num_param_server, _bp_hyper_params.batch_size, dnn->handle());
    data->set_devId(devId);
    dnn->fineTune(data, 500);

#else
    //DMnistData<float> *data = new DMnistData<float>("../data/", DData<float>::Train, 50000, false, dnn->handle());
    //DData<float> *data = new DDummyData<float>(input_dim, 1, handle);
    //DTimitData<float> *data = new DTimitData<float>("/scratch/jxie/", 10000, false, handle);
    //DData<float> *data = new DPatchData<float>("/projects/grail/jxie/paris/", input_dim, 10000, false, handle);
    DData<float> *data = new DRankData<float>("../data/", input_dim, 64, false, handle);
#ifndef DISABLE_GPU
    data->set_devId(devId);
#endif

    //neurons[num_layers-1] = new DTanhNeuron<float>(handle);
    //neurons[num_layers-1] = new DSoftmaxNeuron<float>(_bp_hyper_params.batch_size, handle);
    //neurons[num_layers-1] = new DGMMNeuron<float>(&_bp_hyper_params, 256, output_dim, 0.1, handle);
    //DvMFNeuron<float> *last_neuron = new DvMFNeuron<float>(&_bp_hyper_params, 32, output_dim, 0.2, handle);
    //last_neuron->init(data);
    //neurons[num_layers-1] = last_neuron;
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neurons, &_pt_hyper_params, &_bp_hyper_params, handle);

    if (grad_check) {
        return !dnn->createGradCheck(data);
    }
    if (resuming == -1 && pt_epochs > 0) dnn->pretrain(data, pt_epochs);
    if (resuming != -1) {
        printf("Resuming from %d-th epoch.\n", resuming);
        //std::stringstream ss;
        //ss << resuming - 10;
        fin = fopen(input_path.c_str(), "r");
        if (fin == 0) {
            printf("Error loading: cannot find file %s!\n", input_path.c_str());
            exit(-1);
        }
        dnn->layers()[0]->weight()->samplePrint();
        dnn->load(fin);
        dnn->layers()[0]->weight()->samplePrint();
        fclose(fin);
        _bp_hyper_params.learning_rate *= std::pow(_bp_hyper_params.learning_rate_decay, resuming);
    }else 
        resuming = 0;
    if (exp_name != "")
        fineTuneWithCheckpoint(dnn, data, bp_epochs, 10, path+exp_name, resuming);
    else 
        dnn->fineTune(data, bp_epochs);

#endif
    if (exp_name != "") {
        FILE *fout = fopen(output_path.c_str(), "w");
        dnn->save(fout);
        fclose(fout);
    }
    if (pred_path != "") {
        data->stop();
        data->set_testing(true);
        dnn->test(data, pred_path);
    }
    //DMnistData<float> *test_data;// = new DMnistData<float>("../data", DData<float>::Test, 10000, false, dnn->handle());
    //test_data->set_devId(devId);
    //test_data = new DMnistData<float>("../data", DData<float>::Test, 10000, true, dnn->handle());
    //printf("Testing Error:%f\n", dnn->test(test_data));

    CUDA_CALL(cudaDeviceReset());
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
