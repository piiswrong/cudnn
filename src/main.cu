#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>
#include <time.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

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

    FILE *fin = NULL;
    std::string master_file, log_path, input_path, output_path, pred_path, data_spec_path;
    DDataSpec *data_spec = new DDataSpec;
    int resuming = -1;
    int devId = -1;
    bool grad_check = false;

    int num_layers = 2;
    int hidden_dim = 10;
    //int input_dim = 351, output_dim = 150;
    //int input_dim = 1568, output_dim = 256;
    //int input_dim = 28*28, output_dim = 10;
    int input_dim = 10, output_dim = 10;
    std::string neuron("ReLU");
    float pt_epochs = 0.0;
    int bp_epochs = 200;
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    _pt_hyper_params.idrop_out = false;
    _pt_hyper_params.idrop_rate = 0.2;
    _pt_hyper_params.hdrop_out = false;
    _pt_hyper_params.weight_decay = false;
    _pt_hyper_params.decay_rate = 0.00;
    _pt_hyper_params.momentum = 0.90;
    _pt_hyper_params.learning_rate = 0.01;

    _bp_hyper_params.check_interval = 500;
    _bp_hyper_params.learning_rate = 0.1;
    _bp_hyper_params.learning_rate_decay = 0.00000;
    _bp_hyper_params.idrop_out = false;
    _bp_hyper_params.idrop_rate = 0.2;
    _bp_hyper_params.hdrop_out = false;
    _bp_hyper_params.hdrop_rate= 0.2;
    _bp_hyper_params.momentum = 0.5;
    _bp_hyper_params.max_momentum = 0.90;
    _bp_hyper_params.step_momentum = 0.04;
    _bp_hyper_params.weight_decay = false;
    _bp_hyper_params.decay_rate = 0.01;

    _bp_hyper_params.batch_size = 128;

#ifdef ADMM
    _bp_hyper_params.decay_rate = 0.001;
#endif

    po::options_description generic_opts("Generic options");
    po::options_description hyper_opts("Hyper-parameters");
    po::options_description data_opts("Data set options");
    po::options_description cmdline_opts;
    po::options_description master_file_opts;
    po::positional_options_description p;
    po::variables_map vm;
    try{
        generic_opts.add_options()
            ("help", "print this help massage")
            ("master", po::value<std::string>(&master_file), "path to master file")
            ("data", po::value<std::string>(&data_spec_path), "path to data specification file")
            ("resume,r", po::value<int>(&resuming)->default_value(-1), "resume from n-th epoch")
            ("device,d", po::value<int>(&devId)->default_value(-1), "id of GPU to use")
            ("check-grad,c", po::value<bool>(&grad_check)->default_value(false)->implicit_value(true), "performe numerical gradient checking")
            ("log-path,l", po::value<std::string>(&log_path)->default_value(""), "path to log file")
            ("input-path,i", po::value<std::string>(&input_path), "path to input network parameter file")
            ("output-path,o", po::value<std::string>(&output_path), "path to output network parameter file")
            ("test,t", po::value<std::string>(&pred_path)->implicit_value(""), "performe test and output predictions to file if specified")
            ("verbose,v", po::value<int>(&log_verbosity)->default_value(VERBOSE_NORMAL), "log output verbosity")
        ;

        hyper_opts.add_options()
            ("num-layers", po::value<int>(&num_layers)->default_value(num_layers), "set hyper-paramter value")
            ("hidden-dim", po::value<int>(&hidden_dim)->default_value(hidden_dim), "set hyper-paramter value")
            ("neuron", po::value<std::string>(&neuron)->default_value(neuron), "set hyper-paramter value")
            ("pt-epochs", po::value<float>(&pt_epochs)->default_value(pt_epochs), "set hyper-paramter value")
            ("bp-epochs", po::value<int>(&bp_epochs)->default_value(bp_epochs), "set hyper-paramter value")
        ;
        _pt_hyper_params.registerParams(hyper_opts, "pt");
        _bp_hyper_params.registerParams(hyper_opts, "bp");

        data_spec->registerParams(data_opts);
        
        cmdline_opts.add(generic_opts).add(hyper_opts);
        master_file_opts.add(hyper_opts);
        p.add("master", -1);

        store(po::command_line_parser(argc, argv).
            options(cmdline_opts).positional(p).run(), vm);
        notify(vm);

        if (vm.count("help")) {
            std::cout << cmdline_opts << "\n";
            exit(0);
        }

        if (vm.count("master")) {
            if (!vm.count("log-path")) log_path = master_file+".log";
            if (!vm.count("input-path")) input_path = master_file+".param";
            if (!vm.count("output-path")) output_path = master_file+".param";
            std::ifstream ifs(master_file.c_str());
            if (!ifs) {
                printf("Cannot open master file %s\n", master_file.c_str());
                exit(-1);
            }
            printf("Using master file %s\n", master_file.c_str());
            store(parse_config_file(ifs, master_file_opts), vm);
            notify(vm);
        }
        if (log_path != "") flog = fopen(log_path.c_str(), "w");
        if (vm.count("data")) {
            std::ifstream ifs(data_spec_path.c_str());
            if (!ifs) {
                printf("Cannot open data specification file %s\n", master_file.c_str());
                exit(-1);
            }
            printf("Using data set %s\n", data_spec_path.c_str());
            store(parse_config_file(ifs, data_opts), vm);
            notify(vm);
            input_dim = data_spec->input_dim;
            output_dim = data_spec->output_dim;
            fs::path base(data_spec_path);
            base.remove_filename();
            data_spec->train_data = canonical(data_spec->train_data, base).c_str();
            data_spec->train_label = canonical(data_spec->train_label, base).c_str();
            data_spec->test_data = canonical(data_spec->test_data, base).c_str();
            data_spec->test_label = canonical(data_spec->test_label, base).c_str();
        }

    }catch(std::exception &e) {
        std::cout << "Error: " << e.what() << "\n";
        std::cout << cmdline_opts << "\n";
        std::cout << data_opts << "\n";
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
    if (devId == -1) devId = 0;
#endif
#endif


    cublasHandle_t handle = 0; 
    CUBLAS_CALL(cublasCreate(&handle));


    int *layer_dims = new int[num_layers+1];
    layer_dims[0] = input_dim;
    layer_dims[num_layers] = output_dim;
    for (int i = 1; i < num_layers; i++) layer_dims[i] = hidden_dim;

    DNeuron<float> **neurons = new DNeuron<float>*[num_layers];
    for (int i = 0; i < num_layers-1; i++) {
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
    neurons[num_layers-1] = new DSoftmaxNeuron<float>(_bp_hyper_params.batch_size, handle);
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
    DData<float> *data;
    if (vm.count("data")) {
        data = new DGeneralData<float,float,int>(*data_spec, 10000, true, false, handle);
    }else {
        data = new DPatchData<float>("../data/", input_dim, 2000, false, handle);
        //data = new DRankData<float>("../data/", input_dim, 64, false, handle);
    }
#ifndef DISABLE_GPU
    data->set_devId(devId);
#endif

    //neurons[num_layers-1] = new DTanhNeuron<float>(handle);
    //neurons[num_layers-1] = new DGMMNeuron<float>(&_bp_hyper_params, 256, output_dim, 0.1, handle);
    //DvMFNeuron<float> *last_neuron = new DvMFNeuron<float>(&_bp_hyper_params, 32, output_dim, 0.2, handle);
    neurons[num_layers-1] = new DClusterNeuron<float>(&_bp_hyper_params, 10, output_dim, 0.1, 0.1*output_dim, handle);
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
    }else 
        resuming = 0;
    if (master_file != "")
        fineTuneWithCheckpoint(dnn, data, bp_epochs, 10, master_file, resuming);
    else 
        dnn->fineTune(data, resuming, resuming+bp_epochs);

#endif
    if (output_path != "") {
        FILE *fout = fopen(output_path.c_str(), "w");
        dnn->save(fout);
        fclose(fout);
    }
    if (vm.count("test")) {
        data->stop();
        data->set_testing(true);
        printf("Testing Loss: %f\n", dnn->test(data, pred_path));
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
