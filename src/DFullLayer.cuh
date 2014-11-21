#ifndef DFULLLAYER_CUH
#define DFULLLAYER_CUH

#include <time.h>
#include <common.cuh>
#include <DHyperParams.h>
#include <DOperators.cuh>
#include <DNeuron.cuh>
#include <DLayer.cuh>
#include <kernels.cuh>



template<class T>
class DFullLayer : public DLayer<T> {
protected:
    bool _on_device;
    cublasHandle_t _handle;
    int _input_dim;
    int _output_dim;
    int _layer_id;
    DNeuron<T> *_neuron;
    DHyperParams* _pt_hyper_params;
    DHyperParams* _bp_hyper_params;

    DMatrix<T> *_momentun, *_weight, *_drv, *_act, *_delta, *_mask;
    curandState *_state;
public:
    DFullLayer(int input_dim, int output_dim, int layer_id, DNeuron<T> *neuron, 
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
        _weight->init(DMatrix<T>::Weight|DMatrix<T>::Normal, 0, 1.0/sqrt((T)_weight->ld()));
        _drv = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _drv->init(DMatrix<T>::Zero);
        _act = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _act->init(DMatrix<T>::One);
        _delta = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _delta->init(DMatrix<T>::Zero);
        _mask = new DMatrix<T>(bp_hyper_params->batch_size, _output_dim+1, _handle);
        _mask->init(DMatrix<T>::Zero);
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

    ~DFullLayer() {
        delete _momentun, _weight, _drv, _act, _delta, _mask;
        if (_on_device) {
            CUDA_CALL(cudaFree((void*)_state));
        }
    }

    virtual DMatrix<T> *act() { return _act; }
    virtual DMatrix<T> *delta() { return _delta; }
    virtual DNeuron<T> *neuron() { return _neuron; }
    
    virtual void clearMomentum() {
        _momentun->init(DMatrix<T>::Zero);
    }

    virtual void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
        _drv->update(dev_data, false, _weight, false);
        _neuron->fprop(_act, _drv);
        if (drop_out) 
            hDropout(_act, _neuron->easyDropout()?NULL:_mask, _state, drop_rate, _act->getT(), _act->nrows(), _act->ncols() - 1, _act->ld());
    }
    
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate, bool rectify_weight, bool rectify_bias) {
        assert(!_weight->getT());
        _neuron->bprop(_delta, _drv, _act);
        if (drop_out && !_neuron->easyDropout()) 
            _delta->applyBinary(OpMul<T>(), _mask, _delta->nrows(), _delta->ncols()-1);

        if (delta)
            delta->update(_delta, false, _weight, true, 1.0, 0.0);
        _momentun->update(pre_act, true, _delta, false, -(1.0-mom)*rate/_delta->nrows(), mom);

        if (decay) {
            _momentun->applyBinary(OpScaleAdd<T>(-(1.0-mom)*rate*decay_rate), _weight, _weight->nrows()-1, _weight->ncols());
        }

        _weight->add(_momentun, 1.0, _weight->nelem() - _weight->ld());
        if (rectify_weight && rectify_bias) 
            _weight->applyBinary(OpRectify<T>(), _weight, _weight->nrows(), _weight->ncols()-1);
        else if (rectify_weight) 
            _weight->applyBinary(OpRectify<T>(), _weight, _weight->nrows()-1, _weight->ncols()-1);
        else if (rectify_bias) {
            printf("Error: doesn't support only rectifying bias!\n");
            exit(-1);
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

    DDim4 output_dim() {
        return DDim4(_act->nrows(), 1, 1, _act->ncols()-1);
    }

    virtual void regParams(std::vector<DMatrix<T>*> &X, std::vector<DMatrix<T>*> &dX) {
        X.push_back(new DMatrix<T>(_weight, 0, _weight->ncols() - 1));
        dX.push_back(new DMatrix<T>(_momentun, 0, _momentun->ncols() - 1));
        _neuron->regParams(X, dX);
    }

    void save(FILE *fout) {
        DMatrix<T> *x = _weight;
        x->dev2host();
        fprintf(fout, "%d %d\n", x->ncols() - 1, x->nrows());
        for (int j = 0; j < x->ncols() - 1; j++) {
            for (int k = 0; k < x->nrows(); k++) 
                fprintf(fout, "%lf ", (double)x->getElem(k, j));
            fprintf(fout, "\n");
        }
    }

    void load(FILE *fin) {
        fscanf(fin, "%*d %*d");
        DMatrix<T> *x = _weight;
        for (int j = 0; j < x->ncols() - 1; j++) {
            for (int k = 0; k < x->nrows(); k++) {
                double t;
                fscanf(fin, "%lf", &t);
                x->getElem(k, j) = (T)t;
            }
        }
        x->host2dev();
    }
};


#endif //DFULLLAYER_CUH
