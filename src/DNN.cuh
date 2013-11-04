#ifndef DNN_CUH
#define DNN_CUH

#include <common.cuh>
#include <DLayer.cuh>
#include <DOperators.cuh>
#include <DHyperParams.cuh>


template<class T>
class DNN {
    bool _on_device;
    cublasStatus_t _handle;
    cudaStream_t _stream;
    int _num_layers;
    int *layer_dims;
    DHyperParams _pt_hyper_params;
    DHyperParams _bp_hyper_params;
    DLayer** _layers;
    DMatrix<T>* _delta;


public:
    DNN(int num_layers, int *layer_dims, DNeuron<T> *neurons, 
        const DHyperParams& pt_hyper_params, const DHyperParams bp_hyper_params,
        bool on_device) {
        _on_device = on_device;
        if (_on_device) {
            CUBLAS_CALL(cublasCreate(&_handle));
            CUDA_CALL(cudaStreamCreate(&_stream));
        }else {
            _handle = 0;
        }
        _num_layers = num_layers;
        _pt_hyper_params = pt_hyper_params;
        _bp_hyper_params = bp_hyper_params;

        _layers = new DLayer*[_num_layers];
        for (int i = 0; i < _num_layers; i++) {
            _layers[i] = new DLayer(layer_dims[i], layer_dims[i+1], neurons[i],
                                    &_pt_hyper_params, &_bp_hyper_params, _handle);
        }
        _delta = new DMatrix<T>(_bp_hyper_params->batch_size, layer_dims[_num_layers-1], _handle);
    }

    T trainOnBatch(DMatrix<T>* x, DMatrix<T>* y) {
        fprop(x, _num_layers, _layers);
        return bprop(x, y, _num_layers, _layers);
    }

    void fprop(DMatrix<T>* x, int num_layers, DLayer<T>* layers) {
        layers[0]->fprop(x);
        for (int i = 1; i < num_layers; i++) layers[i]->fprop(layers[i-1]->act());
    }

    T bprop(DMatrix<T>* x, DMatrix<T>* y, int num_layers, DLayer<T>** layers) {
        _delta->applyTenary(OpSub(), layers[num_layers-1]->act, y, _delta->nelem() - _delta->ld()); //TODO:Support other loss.
        T loss = _delta.norm2(_delta->nelem() - _delta->ld())/(_delta->nelem() - _delta->ld());
        DMatrix<T>* d = _delta;
        for (int i = num_layers-1; i > 0; i--) {
            layers[i]->bprop(d, layers[i-1]->act(), _bp_hyper_params->learning_rate, _bp_hyper_params->momentum);
            d = layers[i]->delta();
        }
        layers[0]->bprop(d, x, _bp_hyper_params->learning_rate, _bp_hyper_params->momentum);
        return loss;
    }

};


#endif //DNN_CUH
