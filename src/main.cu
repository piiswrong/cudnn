#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>

int main() {
    cublasHandle_t handle; 
    cublasCreate(&handle);

    int num_layers = 2;
    int layer_dims[] = {28*28, 1023, 10};
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    _bp_hyper_params.batch_size = 10;
    _bp_hyper_params.check_interval = 10;
    DNeuron<float> **neuron = new DNeuron<float>*[num_layers];
    neuron[0] = new DNeuron<float>(handle);
    neuron[1] = new DSoftmaxNeuron<float>(handle);
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neuron, _pt_hyper_params, _bp_hyper_params, handle);
    DMnistData<float> *data = new DMnistData<float>("../data", DData<float>::Train, _bp_hyper_params.batch_size, dnn->handle());
    //DData<float> *data = new DDummyData<float>(10,  handle);
    dnn->fineTune(data, 1);


    return 0;
}
