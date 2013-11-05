#ifndef DLAYER_CUH
#define DLAYER_CUH

#include <time.h>
#include <common.cuh>
#include <DNeuron.cuh>
#include <kernels.cuh>


template<class T>
class DLayer {
    bool _on_device;
    cublasStatus_t _handle;
    int _input_dim;
    int _output_dim;
    DNeuron<T> _neuron;
    DHyperParams* _pt_hyper_params;
    DHyperParams* _bp_hyper_params;

    DMatrix<T>* _momentun, _weight, _drv, _act, _delta;
    curandState *_state;
public:
    DLayer(int input_dim, int output_dim, const DNeuron<T>& neuron, 
           DHyperParams* pt_hyper_params, DHyperParams* bp_hyper_params, handle) {
        _handle = handle;
        if (_handle) {
            _on_device = true;
        }else {
            _on_device = false;
        }
        _input_dim = input_dim;
        _output_dim = output_dim;
        _neuron = neuron;
        _pt_hyper_params = pt_hyper_params;
        _bp_hyper_params = bp_hyper_params;
        
        _momentun = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        _momentun->init(DMatrixInit::Zero);
        _weight = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        _weight->init(DMatrixInit::Weight|DMatrixInit::Normal, 0.0, 1/sqrt((T)_weight));
        _drv = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _act = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        if (_on_device) {
            CUDA_CALL(cudaMalloc((void**)&_state, _act->nelem()*sizeof(curandState)));
            dim3 grid((_act->nelem()-1)/BLOCK_SIZE+1);
            dim3 block(BLOCK_SIZE);
            kSetupCurand<<<grid, block>>>(_state, _act->nelem(), time());
        }
    }

    T* delta() { return _delta; }
    T* act() { return _act; }

    void fprop(DMatrix<T>* dev_data, float drop_rate = 0.0) {
        _drv->update(*dev_data, *_weight);
        neuron.fprop(_act, _drv);
        if (drop_rate > 0.0) {
            dim3 grid((_act->nelem()-1)/BLOCK_SIZE+1);
            dim3 block(BLOCK_SIZE);
            kDropout<T><<<grid, block>>>(_act->dev_data(), _state, _act->nelem() - _act->ld(), drop_rate);
        }
    }
    
    T bprob(DMatrix<T>* delta, DMatrix<T>* pre_act, T rate, T mom) {
        neuron.bprop(delta, _drv, _act);
        _momentun.update(pre_act, true, delta, false, rate/delta.nrows(), mom);
        _delta.update(delta, false, _weight, true, 1.0, 0.0);
        _weight.add(_momentun, 1.0, _weight->nelem() - _weight->ld());
    }
};


#endif //DLAYER_CUH
