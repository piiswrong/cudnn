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
    int *_layer_dims;
    DHyperParams _pt_hyper_params;
    DHyperParams _bp_hyper_params;
    DLayer<T>** _layers;
    DMatrix<T>* _delta;
    curandState *_state;

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
        _layer_dims = layer_dims;
        _pt_hyper_params = pt_hyper_params;
        _bp_hyper_params = bp_hyper_params;

        _layers = new DLayer<T>*[_num_layers];
        for (int i = 0; i < _num_layers; i++) {
            _layers[i] = new DLayer<T>(layer_dims[i], layer_dims[i+1], neurons[i],
                                    &_pt_hyper_params, &_bp_hyper_params, _handle);
        }
        _delta = new DMatrix<T>(_bp_hyper_params.batch_size, layer_dims[_num_layers]+1, _handle);
        _delta->init(DMatrix<T>::Zero);
        if (_on_device) {
            int nstate = (_layer_dims[0]+1)*_bp_hyper_params.batch_size;
            CUDA_CALL(cudaMalloc((void**)&_state, nstate*sizeof(curandState)));
            dim3 grid((nstate-1)/BLOCK_SIZE+1);
            dim3 block(BLOCK_SIZE);
            kSetupCurand<<<grid, block>>>(_state, nstate, time(0));
        }
    }

    cublasHandle_t handle() { return _handle; }

    T trainOnBatch(DMatrix<T>* x, DMatrix<T>* y) {
        fprop(x, _num_layers, _layers, _bp_hyper_params.idrop_out, _bp_hyper_params.hdrop_out,
                _bp_hyper_params.idrop_rate, _bp_hyper_params.hdrop_rate);
        return bprop(x, y, _num_layers, _layers);
    }

    virtual void fprop(DMatrix<T>* x, int num_layers, DLayer<T>** layers, bool idrop_out = false,
                        bool hdrop_out = false, float idrop_rate = 0.0, float hdrop_rate = 0.0) {
        if (idrop_out) {
            dim3 grid((x->nelem() - x->ld() - 1)/BLOCK_SIZE+1, 1, 1);
            dim3 block(BLOCK_SIZE, 1, 1);
            if ((x->nelem() - x->ld())%BLOCK_SIZE == 0)
                kDropout<T, true><<<grid, block>>>(x->dev_data(), _state, x->nelem() - x->ld(), idrop_rate);
            else
                kDropout<T, false><<<grid, block>>>(x->dev_data(), _state, x->nelem() - x->ld(), idrop_rate);
        }
        layers[0]->fprop(x, num_layers > 1 && hdrop_out, hdrop_rate);
        for (int i = 1; i < num_layers; i++) 
            layers[i]->fprop(layers[i-1]->act(), i < num_layers - 1 && hdrop_out, hdrop_rate);
    }

    virtual T bprop(DMatrix<T>* x, DMatrix<T>* y, int num_layers, DLayer<T>** layers) {
        //_delta->applyTenary(OpSub(), layers[num_layers-1]->act, y, _delta->nelem() - _delta->ld()); //TODO:Support other loss.
        layers[num_layers-1]->neuron()->initDelta(_delta, layers[num_layers-1]->act(), y);
        DMatrix<T>* d = _delta;
        for (int i = num_layers-1; i > 0; i--) {
            layers[i]->bprop(d, layers[i-1]->act(), _bp_hyper_params.learning_rate, _bp_hyper_params.momentum);
            d = layers[i]->delta();
        }
        layers[0]->bprop(d, x, _bp_hyper_params.learning_rate, _bp_hyper_params.momentum);
//        layers[num_layers-1]->neuron()->computeLoss(_delta, layers[num_layers-1]->act(), y);
        return layers[num_layers-1]->neuron()->getLoss();
    }
    
    void fineTune(DData<T>* data, int total_epochs) {
        data->start();
        int iperEpoch = 1280;//data->instancesPerEpoch();
        DMatrix<T> *x, *y;
        int nEpoch = 1;
        int nInstance = 0;
        T error = 0.0;
        int lastCheck = 0;
        while ( nEpoch <= total_epochs ) {
            data->getData(x, y, _bp_hyper_params.batch_size);
            //printf("%d:%d\n", nEpoch, nInstance);
            cudaThreadSynchronize();
            error += trainOnBatch(x, y);
            nInstance += _pt_hyper_params.batch_size;
            while (nEpoch*iperEpoch <= nInstance) {
                nEpoch++;
                _bp_hyper_params.learning_rate *= _bp_hyper_params.learning_rate_decay;
                _bp_hyper_params.momentum += _bp_hyper_params.step_momentum;
                if (_bp_hyper_params.momentum > _bp_hyper_params.max_momentum)
                    _bp_hyper_params.momentum = _bp_hyper_params.max_momentum;
            }
            lastCheck += _bp_hyper_params.batch_size;
            if (lastCheck >= _bp_hyper_params.check_interval) {
#ifndef NDEBUG
/*                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = x;//_layers[i]->delta();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }*/
                DMatrix<T> *m = y;//_layers[i]->delta();
                m->dev2host();
                for (int r = 0; r < 10; r++) {
                    for (int c = 0; c < 10; c++) {
                        printf("%+1.3f ", (float)m->host_data()[r*10+c]);
                    }
                    printf("\n");
                }
                printf("\n");
                

                m = _delta;//_layers[i]->delta();
                m->dev2host();
                for (int r = 0; r < 10; r++) {
                    for (int c = 0; c < m->fd(); c++) {
                        printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                    }
                    printf("\n");
                }
                printf("\n");
                
              /*  for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = _layers[i]->drv();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = _layers[i]->act();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = _layers[i]->weight();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }*/
#endif
                printf("\nEpoch: %d\nInstance: %d\nError: %f\n", nEpoch, nInstance%iperEpoch, error);
                lastCheck = 0;
                error = 0.0;
            }
        }
        
    }

};


#endif //DNN_CUH
