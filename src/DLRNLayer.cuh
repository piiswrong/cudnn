#ifndef DLRNLAYER_CUH
#define DLRNLAYER_CUH

#include <common.cuh>
#include <DLayer.cuh>

template<class T>
class DLRNLayer : public DLayer<T> {
    cublasHandle_t _handle;
    DDim4 _output_dim;
    int _kernel_size;
    T _alpha, _beta;
    DMatrix<T> *_act, *_scale, *_delta;

    dim3 _block, _grid;

public:
    DLRNLayer(DDim4 input_dim, T alpha, T beta, int kernel_size, cublasHandle_t handle) : 
        _block(min(WARP_SIZE, input_dim.w), min(WARP_SIZE, input_dim.h)),
        _grid((input_dim.w-1)/_block.x+1, (input_dim.h-1)/_block.y+1, input_dim.n) {
        _handle = handle;
        _output_dim = input_dim;
        _kernel_size = kernel_size;
        if (_kernel_size > input_dim.c) _kernel_size = input_dim.c;

        DDim4 &id = input_dim;
        _act = new DMatrix<T>(id.c*id.h*id.w+1, id.n, _handle);
        _act->init(DMatrix<T>::One);
        _act->setT();
        _scale = new DMatrix<T>(id.c*id.h*id.w+1, id.n, _handle);
        _scale->init(DMatrix<T>::One);
        _scale->setT();
        _delta = new DMatrix<T>(id.c*id.h*id.w+1, id.n, _handle);
        _delta->init(DMatrix<T>::One);
        _delta->setT();
    }

    DDim4 output_dim() { return _output_dim; }
    virtual DMatrix<T> *act() { return _act; }
    virtual DMatrix<T> *delta() { return _delta; }
    virtual DNeuron<T> *neuron() { return NULL; }

    virtual void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
        kLRNForward<<<_grid, _block>>>(dev_data->dev_data(), _act->dev_data(), _scale->dev_data(), _alpha/_kernel_size, -_beta, _output_dim.n, _output_dim.c, _output_dim.h, _output_dim.w, _kernel_size, _act->ld());
        CUDA_KERNEL_CHECK();
    }

    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate, bool rectify_weight, bool rectify_bias) {
        if (delta) {
            kLRNBackward<<<_grid, _block>>>(delta->dev_data(), pre_act->dev_data(), _act->dev_data(), _scale->dev_data(), _delta->dev_data(), _alpha/_kernel_size, -_beta, 2*_beta*_alpha/_kernel_size, _output_dim.n, _output_dim.c, _output_dim.h, _output_dim.w, _kernel_size, _act->ld());
            CUDA_KERNEL_CHECK();
        }
    }

};

#endif //DLRNLAYER_CUH
