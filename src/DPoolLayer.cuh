#ifndef DPOOLLAYER_CUH
#define DPOOLLAYER_CUH

#include <cudnn.h>
#include <common.cuh>
#include <DLayer.cuh>

template<class T>
class DPoolLayer : public DLayer<T> {
protected:
    cublasHandle_t _handle;
    cudnnHandle_t _cudnn_handle;
    cudnnTensor4dDescriptor_t _input_desc, _output_desc; 
    cudnnPoolingDescriptor_t _pool_desc;
    DDim4 _input_dim, _output_dim;

    DMatrix<T> *_act, *_delta; 


    cudnnDataType_t dtype() { 
        return CUDNN_DATA_FLOAT;
    }
public:
    DPoolLayer(DDim4 input_dim, int kernel_size, int stride, cublasHandle_t handle, cudnnHandle_t cudnn_handle) {
        _handle = handle;
        _cudnn_handle = cudnn_handle;

        _input_dim = input_dim;
        DDim4 &id = _input_dim;
        CUDNN_CALL(cudnnCreateTensor4dDescriptor(&_input_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptorEx(_input_desc, dtype(), id.n, id.c, id.h, id.w, id.c*id.h*id.w+1, id.h*id.w, id.w, 1));

        CUDNN_CALL(cudnnCreatePoolingDescriptor(&_pool_desc));
        CUDNN_CALL(cudnnSetPoolingDescriptor(_pool_desc, CUDNN_POOLING_MAX, kernel_size, kernel_size, stride, stride));

        DDim4 &od = _output_dim;
        od = _input_dim;
        od.w = (od.w - kernel_size)/stride + 1;
        od.h = (od.h - kernel_size)/stride + 1;

        CUDNN_CALL(cudnnCreateTensor4dDescriptor(&_output_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptorEx(_output_desc, dtype(), od.n, od.c, od.h, od.w, od.c*od.h*od.w+1, od.h*od.w, od.w, 1));

        _act = new DMatrix<T>(od.c*od.h*od.w+1, od.n, _handle);
        _act->init(DMatrix<T>::One);
        _act->setT();
        _delta= new DMatrix<T>(od.c*od.h*od.w+1, od.n, _handle);
        _delta->init(DMatrix<T>::One);
        _delta->setT();
    }

    DDim4 output_dim() { return _output_dim; }
    virtual DMatrix<T> *act() { return _act; }
    virtual DMatrix<T> *delta() { return _delta; }
    virtual DNeuron<T> *neuron() { return NULL; }

    
    virtual void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
        CUDNN_CALL(cudnnPoolingForward(_cudnn_handle, _pool_desc, _input_desc, dev_data->dev_data(), _output_desc, _act->dev_data()));
    }

    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate, bool rectify_weight, bool rectify_bias) {
        if (delta) CUDNN_CALL(cudnnPoolingBackward(_cudnn_handle, _pool_desc, _output_desc, _act->dev_data(), _output_desc, _delta->dev_data(), _input_desc, pre_act->dev_data(), _input_desc, delta->dev_data()));
    }
};

#endif //DPOOLLAYER_CUH
