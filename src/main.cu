#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>

int main() {
    cublasHandle_t handle; 
    cublasCreate(&handle);

    int num_layers = 3;
    int layer_dims[] = {28*28, 2047, 2047, 10};
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    //_bp_hyper_params.batch_size = 10;
    _bp_hyper_params.check_interval = 10000;
    _bp_hyper_params.learning_rate = 1.0;
//    _bp_hyper_params.idrop_out = false;
//    _bp_hyper_params.hdrop_out = false;
    _bp_hyper_params.momentum = 0.0;
    _bp_hyper_params.max_momentum = 0.0;

    //_bp_hyper_params.sparseInit = true;
    DNeuron<float> **neuron = new DNeuron<float>*[num_layers];
    neuron[0] = new DOddrootNeuron<float>(handle);
    neuron[1] = new DOddrootNeuron<float>(handle);
    neuron[2] = new DSoftmaxNeuron<float>(_bp_hyper_params.batch_size, handle);
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neuron, _pt_hyper_params, _bp_hyper_params, handle);
    DMnistData<float> *data = new DMnistData<float>("../data", DData<float>::Train, 50000, false, dnn->handle());
    //DData<float> *data = new DDummyData<float>(10,  handle);
    dnn->fineTune(data, 100);

    DMnistData<float> *test_data;// = new DMnistData<float>("../data", DData<float>::Test, 10000, false, dnn->handle());
    //dnn->fineTune(test_data, 1);
    test_data = new DMnistData<float>("../data", DData<float>::Test, 10000, true, dnn->handle());
    printf("Testing Error:%f\n", dnn->test(test_data));

    cudaDeviceReset();
    return 0;
}
