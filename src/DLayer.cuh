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
    int _layer_id;
    DNeuron<T> *_neuron;
    DHyperParams* _pt_hyper_params;
    DHyperParams* _bp_hyper_params;

    DMatrix<T> *_momentun, *_weight, *_drv, *_act, *_delta, *_mask;
#ifdef ADMM
    DMatrix<T> *_z, *_u, *_buf;
#endif
#ifdef DOWN_POUR_SGD
    DMatrix<T> *_grad;
#endif
    curandState *_state;
public:
    DLayer(int input_dim, int output_dim, int layer_id, DNeuron<T> *neuron, 
           DHyperParams* pt_hyper_params, DHyperParams* bp_hyper_params, cublasHandle_t handle) {
        _handle = handle;
        if (_handle) {
            _on_device = true;
        }else {
            _on_device = false;
        }
        _input_dim = input_dim;
        _output_dim = output_dim;
        _layer_id = layer_id;
        _neuron = neuron;
        _pt_hyper_params = pt_hyper_params;
        _bp_hyper_params = bp_hyper_params;
        
        _momentun = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        _momentun->init(DMatrix<T>::Zero);
        _weight = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
#ifdef DOWN_POUR_SGD
        _weight->mpiCreateWin();
#endif
//        if (_bp_hyper_params->sparseInit) {
//            _weight->init(DMatrix<T>::Weight|DMatrix<T>::Uniform|DMatrix<T>::ColSparse, -1.0/sqrt((T)_weight->ld()), 1.0/sqrt((T)_weight->ld()));
//        }else {
            _weight->init(DMatrix<T>::Weight|DMatrix<T>::Uniform, -1.0/sqrt((T)_weight->ld()), 1.0/sqrt((T)_weight->ld()));
//        }
        _drv = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _drv->init(DMatrix<T>::Zero);
        _act = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _act->init(DMatrix<T>::One);
        _delta = new DMatrix<T>(bp_hyper_params->batch_size, _input_dim+1, _handle);
        _delta->init(DMatrix<T>::Zero);
        _mask = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _mask->init(DMatrix<T>::Zero);
#ifdef ADMM
        _z = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        _z->init(DMatrix<T>::Zero);
        _u = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        _u->init(DMatrix<T>::Zero);
        _buf = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
        _buf->init(DMatrix<T>::Zero);
#endif

#ifdef DOWN_POUR_SGD
        if (mpi_world_rank < sgd_num_param_server) {
            _grad = new DMatrix<T>(_input_dim+1, _output_dim+1, _handle);
            _grad->init(DMatrix<T>::Zero);
        }
#endif

#ifndef DISABLE_GPU
        if (_on_device) {
            CUDA_CALL(cudaMalloc((void**)&_state, _act->nelem()*sizeof(curandState)));
            dim3 grid((_act->nelem()-1)/BLOCK_SIZE+1);
            dim3 block(BLOCK_SIZE);
            kSetupCurand<<<grid, block>>>(_state, _act->nelem(), time(0));
            CUDA_KERNEL_CHECK();
        }
#endif
    }

    ~DLayer() {
        delete _momentun, _weight, _drv, _act, _delta, _mask;
#ifdef ADMM
        delete _z, _u, _buf;
#endif
#ifdef DOWN_POUR_SGD
        delete _grad;
#endif
        if (_on_device) {
            CUDA_CALL(cudaFree((void*)_state));
        }
    }

    DMatrix<T> *delta() { return _delta; }
    DMatrix<T> *drv() { return _drv; }
    DMatrix<T> *act() { return _act; }
    DMatrix<T> *weight() { return _weight; }
    DNeuron<T> *neuron() { return _neuron; }
    curandState*state() { return _state; }

    
    void clearMomentum() {
        _momentun->init(DMatrix<T>::Zero);
    }

#ifdef ADMM
    void ADMM_reduce() {
        _z->applyTenary(OpAdd<T>(), _weight, _u, _weight->nrows(), _weight->ncols());
        _z->mpiAllReduce(_z, MPI_SUM);
        _z->applyBinary(OpScale<T>(1.0/mpi_world_size), _z, _z->nrows(), _z->ncols());
        _u->applyTenary(OpSubEqu<T>(), _weight, _z, _weight->nrows(), _weight->ncols());
        _buf->applyTenary(OpSub<T>(), _u, _z, _buf->nrows(), _buf->ncols());
        _weight->mpiAllReduce(_weight, MPI_SUM);
        _weight->applyBinary(OpScale<T>(1.0/mpi_world_size), _weight, _weight->nrows(), _weight->ncols());
    }
#endif

    void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
#ifdef DOWN_POUR_SGD
        _weight->mpiGet(0);
        //printf("%f\n", _weight->getElem(0,0));
#endif
        _drv->update(dev_data, false, _weight, false);
        _neuron->fprop(_act, _drv);
        if (drop_out) 
            hDropout(_act, _neuron->easyDropout()?NULL:_mask, _state, drop_rate, _act->getT(), _act->nrows(), _act->ncols() - 1, _act->ld());
    }
    
    void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate) {
        assert(!_weight->getT());
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) {
#endif
            _neuron->bprop(delta, _drv, _act);
            if (drop_out && !_neuron->easyDropout()) 
                delta->applyBinary(OpMul<T>(), _mask, delta->nrows(), delta->ncols()-1);

            _delta->update(delta, false, _weight, true, 1.0, 0.0);
#ifdef DOWN_POUR_SGD
            _momentun->update(pre_act, true, delta, false, -(1.0-mom)*rate/delta->nrows(), 0.0);
#else
            _momentun->update(pre_act, true, delta, false, -(1.0-mom)*rate/delta->nrows(), mom);
#endif

#ifdef ADMM
            _momentun->applyTenary(OpADMMDecay<T>(-(1.0-mom)*rate*decay_rate), _weight, _buf, _weight->nrows(), _weight->ncols());
#else
            if (decay) {
                _momentun->applyBinary(OpScaleAdd<T>(-(1.0-mom)*rate*decay_rate), _weight, _weight->nrows()-1, _weight->ncols());
            }
#endif

#ifdef DOWN_POUR_SGD
            _momentun->mpiSend(0, _layer_id);
        }else {
            _grad->mpiRecv(MPI_ANY_SOURCE, _layer_id);
            _momentun->applyBinary(OpDPSGDMom<T>(mom), _grad, _weight->nrows(), _weight->ncols());
            T tt = _weight->getElem(0,0);
            _weight->add(_momentun, 1.0, _weight->nelem() - _weight->ld());
            //printf("%f+%f=%f\n", tt, _momentun->getElem(0,0), _weight->getElem(0,0));
        }
#else
        _weight->add(_momentun, 1.0, _weight->nelem() - _weight->ld());
#endif
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
