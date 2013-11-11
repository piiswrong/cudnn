#ifndef DNN_CUH
#define DNN_CUH

#include <common.cuh>
#include <DData.cuh>
#include <DLayer.cuh>
#include <DOperators.cuh>
#include <DHyperParams.cuh>


template<class T>
class DNN {
    bool _on_device;
    cublasHandle_t _handle;
    int _num_layers;
    int *layer_dims;
    DHyperParams _pt_hyper_params;
    DHyperParams _bp_hyper_params;
    DLayer<T>** _layers;
    DMatrix<T>* _delta;


public:
    DNN(int num_layers, int *layer_dims, DNeuron<T> **neurons, 
        DHyperParams& pt_hyper_params, DHyperParams& bp_hyper_params,
        cublasHandle_t handle = 0) {
        _handle = handle;
        if (_handle != 0) {
            _on_device = true;
        }else {
            _on_device = false;
        }
        _num_layers = num_layers;
        _pt_hyper_params = pt_hyper_params;
        _bp_hyper_params = bp_hyper_params;

        _layers = new DLayer<T>*[_num_layers];
        for (int i = 0; i < _num_layers; i++) {
            _layers[i] = new DLayer<T>(layer_dims[i], layer_dims[i+1], neurons[i],
                                    &_pt_hyper_params, &_bp_hyper_params, _handle);
        }
        _delta = new DMatrix<T>(_bp_hyper_params.batch_size, layer_dims[_num_layers-1], _handle);
    }

    cublasHandle_t handle() { return _handle; }

    T trainOnBatch(DMatrix<T>* x, DMatrix<T>* y) {
        fprop(x, _num_layers, _layers, _bp_hyper_params.hdrop_rate);
        return bprop(x, y, _num_layers, _layers);
    }

    virtual void fprop(DMatrix<T>* x, int num_layers, DLayer<T>** layers, float drop_rate = 0.0) {
        layers[0]->fprop(x, (0 == num_layers - 1) ? 0.0 : drop_rate);
        for (int i = 1; i < num_layers; i++) layers[i]->fprop(layers[i-1]->act(), (i == num_layers - 1) ? 0.0 : drop_rate);
    }

    virtual T bprop(DMatrix<T>* x, DMatrix<T>* y, int num_layers, DLayer<T>** layers) {
        //_delta->applyTenary(OpSub(), layers[num_layers-1]->act, y, _delta->nelem() - _delta->ld()); //TODO:Support other loss.
        layers[num_layers-1]->neuron()->initDelta(_delta, layers[num_layers-1]->act(), y);
        layers[num_layers-1]->neuron()->computeLoss(_delta, layers[num_layers-1]->act(), y);
        DMatrix<T>* d = _delta;
        for (int i = num_layers-1; i > 0; i--) {
            layers[i]->bprop(d, layers[i-1]->act(), _bp_hyper_params.learning_rate, _bp_hyper_params.momentum);
            d = layers[i]->delta();
        }
        layers[0]->bprop(d, x, _bp_hyper_params.learning_rate, _bp_hyper_params.momentum);
        return layers[num_layers-1]->neuron()->getLoss();
    }
    
    void fineTune(DData<T>* data, int total_epochs) {
        data->start();
        int iperEpoch = data->instancesPerEpoch();
        DMatrix<T> *x, *y;
        int nEpoch = 0;
        int nInstance = 0;
        T error = 0.0;
        int lastCheck = 0;
        while ( nEpoch < total_epochs ) {
            data->getData(x, y, _bp_hyper_params.batch_size);
            cudaThreadSynchronize();
            error += trainOnBatch(x, y);
            nInstance += _pt_hyper_params.batch_size;
            while (nEpoch*iperEpoch < nInstance) {
                nEpoch++;
                _bp_hyper_params.learning_rate *= _bp_hyper_params.learning_rate_decay;
                _bp_hyper_params.momentum += _bp_hyper_params.step_momentum;
                if (_bp_hyper_params.momentum > _bp_hyper_params.max_momentum)
                    _bp_hyper_params.momentum = _bp_hyper_params.max_momentum;
            }
            lastCheck += _bp_hyper_params.batch_size;
            if (lastCheck >= _bp_hyper_params.check_interval) {
                printf("\nEpoch: %d\nInstance: %d\nError: %f\n", nEpoch, nInstance%nEpoch, error);
                lastCheck = 0;
                error = 0.0;
            }
        }
        
    }

};


#endif //DNN_CUH
