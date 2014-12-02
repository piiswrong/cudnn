#ifndef DLRNLAYER_CUH
#define DLRNLAYER_CUH

#include <common.cuh>
#include <DLayer.cuh>

template<class T>
class DLRNLayer : public DLayer<T> {
    cublasHandle_t _handle;
    DDim4 _output_dim;
    DMatrix<T> *_act, *_delta;

public:
    DLRNLayer(DDim4 input_dim, int kernel_size, cublasHandle_t handle) {
        _handle = handle;
        _output_dim = input_dim;

        DDim4 &id = input_dim;
        _act = new DMatrix<T>(id.c*id.h*id.w+1, id.n, _handle);
        _act->init(DMatrix<T>::One);
        _act->setT();
        _delta = new DMatrix<T>(id.c*id.h*id.w+1, id.n, _handle);
        _delta->init(DMatrix<T>::One);
        _delta->setT();
    }

    DDim4 output_dim() { return _output_dim; }
    virtual DMatrix<T> *act() { return _act; }
    virtual DMatrix<T> *delta() { return _delta; }

    virtual void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
    }

    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate, bool rectify_weight, bool rectify_bias) {
    }

};

#endif //DLRNLAYER_CUH
