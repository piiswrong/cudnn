#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>

int main() {
    cublasHandle_t handle; 
    cublasCreate(&handle);

    int num_layers = 2;
    int layer_dims[] = {7, 7, 7};
    DHyperParams _bp_hyper_params, _pt_hyper_params;
    _bp_hyper_params.batch_size = _pt_hyper_params.batch_size = 1;
    DNeuron<float> **neuron = new DNeuron<float>*[2];
    neuron[0] = new DReLUNeuron<float>(handle);
    neuron[1] = new DSoftmaxNeuron<float>(handle);
    
    DNN<float> *dnn = new DNN<float>(num_layers, layer_dims, neuron, _pt_hyper_params, _bp_hyper_params, handle);
    //DMnistData<float> *data = new DMnistData<float>("../data", DData<float>::Train, dnn->handle());
    DDummyData<float> *data = new DDummyData<float>(8, handle);
    dnn->fineTune(data, 1000000);


    return 0;
}
