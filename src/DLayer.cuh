#ifndef DLAYER_CUH
#define DLAYER_CUH

#include <time.h>
#include <common.cuh>
#include <DHyperParams.cuh>
#include <DOperators.cuh>
#include <DNeuron.cuh>
#include <kernels.cuh>


template<class T>
class DLayer {
    bool _on_device;
    cublasHandle_t _handle;
    int _input_dim;
    int _output_dim;
    DNeuron<T> *_neuron;
    DHyperParams* _pt_hyper_params;
    DHyperParams* _bp_hyper_params;

    DMatrix<T> *_momentun, *_weight, *_drv, *_act, *_delta, *_mask;
    curandState *_state;
public:
    DLayer(int input_dim, int output_dim, DNeuron<T> *neuron, 
           DHyperParams* pt_hyper_params, DHyperParams* bp_hyper_params, cublasHandle_t handle) {
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
        _momentun->init(DMatrix<T>::Zero);
        _weight = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        if (_bp_hyper_params->sparseInit) {
            _weight->init(DMatrix<T>::Weight|DMatrix<T>::Uniform|DMatrix<T>::ColSparse, -1.0/sqrt((T)_weight->ld()), 1.0/sqrt((T)_weight->ld()));
        }else {
            _weight->init(DMatrix<T>::Weight|DMatrix<T>::Uniform, -1.0/sqrt((T)_weight->ld()), 1.0/sqrt((T)_weight->ld()));
        }
        _drv = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _drv->init(DMatrix<T>::Zero);
        _act = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _act->init(DMatrix<T>::One);
        _delta = new DMatrix<T>(bp_hyper_params->batch_size, _input_dim+1, _handle);
        _delta->init(DMatrix<T>::Zero);
        _mask = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _mask->init(DMatrix<T>::Zero);
#ifndef DISABLE_GPU
        if (_on_device) {
            CUDA_CALL(cudaMalloc((void**)&_state, _act->nelem()*sizeof(curandState)));
            dim3 grid((_act->nelem()-1)/BLOCK_SIZE+1);
            dim3 block(BLOCK_SIZE);
            kSetupCurand<<<grid, block>>>(_state, _act->nelem(), time(0));
        }
#endif
    }

    DMatrix<T> *delta() { return _delta; }
    DMatrix<T> *drv() { return _drv; }
    DMatrix<T> *act() { return _act; }
    DMatrix<T> *weight() { return _weight; }
    DNeuron<T> *neuron() { return _neuron; }

    
    void clearMomentum() {
        _momentun->init(DMatrix<T>::Zero);
    }

    void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
        _drv->update(dev_data, false, _weight, false);
        _neuron->fprop(_act, _drv);
        if (drop_out) 
            hDropout(_act->dev_data(), _neuron->easyDropout()?NULL:_mask->dev_data(), _state, drop_rate, _act->getT(), _act->nrows(), _act->ncols() - 1, _act->ld());
    }
    
    void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate) {
        assert(!_weight->getT());
        _neuron->bprop(delta, _drv, _act);
        if (drop_out && !_neuron->easyDropout()) 
            _delta->applyBinary(OpMul<T>(), _mask, _delta->nrows(), _delta->ncols()-1);

        _delta->update(delta, false, _weight, true, 1.0, 0.0);
        _momentun->update(pre_act, true, delta, false, -(1.0-mom)*rate/delta->nrows(), mom);
        if (decay) {
#ifndef DISABLE_GPU
            if (_on_device) {
                int ld = _weight->ld();
                int fd = _weight->fd(); 
                dim3 grid((ld-1)/TILE_DIM+1, (fd-1)/TILE_DIM+1, 1);
                dim3 block(TILE_DIM, BLOCK_ROWS, 1);
                kWeightUpdate<T><<<grid, block>>>(_weight->dev_data(), _momentun->dev_data(), (1.0-decay_rate), ld, fd);
            }else 
#endif
            {
                int m = _weight->nrows(), n = _weight->ncols();
                T factor = 1.0-decay_rate;
                for (int j = 0; j < n - 1; j++) {
                    for (int i = 0; i < m - 1; i++) {
                        _weight->getElem(i,j) = _weight->getElem(i,j)*factor + _momentun->getElem(i,j);
                    }
                }
                for (int i = 0; i < n - 1; i++)
                    _weight->getElem(m-1,j) = _weight->getElem(m-1,j)*factor + _momentun->getElem(m-1,j);
            }
        }else {
            _weight->add(_momentun, 1.0, _weight->nelem() - _weight->ld());
        }
    }


    void scaleWeight(float scale) {
        _weight->dev2host();
        T* data = _weight->host_data();
        int ld = _weight->ld(), fd = _weight->fd();
        for (int i = 0; i < fd - 1; i++)
            for (int j = 0; j < ld - 1; j++) 
                data[i*ld + j] *= scale;
        _weight->host2dev();
    }

    void save(FILE *fout) {
    }
};


#endif //DLAYER_CUH
