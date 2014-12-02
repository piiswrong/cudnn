#ifndef DCONVLAYER_CUH
#define DCONVLAYER_CUH

#include <cudnn.h>
#include <common.cuh>
#include <DLayer.cuh>

template<class T>
class DConvLayer : public DLayer<T> {
protected:
    DNeuron<T> *_neuron;
    cublasHandle_t _handle;
    cudnnHandle_t _cudnn_handle;
    cudnnTensor4dDescriptor_t _input_desc, _output_desc, _bias_desc; 
    cudnnFilterDescriptor_t _filter_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
    DDim4 _input_dim, _output_dim, _filter_dim;

    DMatrix<T> *_filter, *_bias, *_grad_filter, *_grad_bias, *_mom_filter, *_mom_bias;
    DMatrix<T> *_drv, *_act, *_delta; 

    cudnnDataType_t dtype() { 
        return CUDNN_DATA_FLOAT;
    }

    void initFilter() {
        _filter->init(DMatrix<T>::Normal, 0.0, 0.01);
    }

public:
    class OpMom {
        T _alpha, _beta;
    public:
        OpMom(T alpha, T beta) : _alpha(alpha), _beta(beta) {}
        HOSTDEVICE T operator() (T x, T y) {
            return _alpha*x + _beta*y;
        }
    };

    DConvLayer(DDim4 input_dim, int num_output, int kernel_size, int stride, int pad, DNeuron<T> *neuron, cublasHandle_t handle, cudnnHandle_t cudnn_handle) {
        _neuron = neuron;
        _handle = handle;
        _cudnn_handle = cudnn_handle;

        _input_dim = input_dim;
        DDim4 &id = _input_dim;
        CUDNN_CALL(cudnnCreateTensor4dDescriptor(&_input_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptorEx(_input_desc, dtype(), id.n, id.c, id.h, id.w, id.c*id.h*id.w+1, id.h*id.w, id.w, 1));
        
        DDim4 &fd = _filter_dim;
        fd.n = num_output;
        fd.c = input_dim.c;
        fd.h = fd.w = kernel_size;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&_filter_desc));
        CUDNN_CALL(cudnnSetFilterDescriptor(_filter_desc, dtype(), fd.n, fd.c, fd.h, fd.w));
        _filter = new DMatrix<T>(fd.c*fd.h*fd.w, fd.n, _handle);
        initFilter();
        _mom_filter = new DMatrix<T>(fd.c*fd.h*fd.w, fd.n, _handle);
        _mom_filter->init(DMatrix<T>::Zero);
        _grad_filter = new DMatrix<T>(fd.c*fd.h*fd.w, fd.n, _handle);

        CUDNN_CALL(cudnnCreateTensor4dDescriptor(&_bias_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(_bias_desc, CUDNN_TENSOR_NCHW, dtype(), 1, num_output, 1, 1));
        _bias = new DMatrix<T>(num_output, 1, _handle);
        _bias->init(DMatrix<T>::Zero);
        _mom_bias = new DMatrix<T>(num_output, 1, _handle);
        _mom_bias->init(DMatrix<T>::Zero);
        _grad_bias = new DMatrix<T>(num_output, 1, _handle);

        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&_conv_desc));
        CUDNN_CALL(cudnnSetConvolutionDescriptor(_conv_desc, _input_desc, _filter_desc, pad, pad, stride, stride, 1, 1, CUDNN_CONVOLUTION));

        DDim4 &od = _output_dim;
        CUDNN_CALL(cudnnGetOutputTensor4dDim(_conv_desc, CUDNN_CONVOLUTION_FWD, &od.n, &od.c, &od.h, &od.w));

        CUDNN_CALL(cudnnCreateTensor4dDescriptor(&_output_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptorEx(_output_desc, dtype(), od.n, od.c, od.h, od.w, od.c*od.h*od.w+1, od.h*od.w, od.w, 1));

        _act = new DMatrix<T>(od.c*od.h*od.w+1, od.n, _handle);
        _act->init(DMatrix<T>::One);
        _act->setT();
        _drv = new DMatrix<T>(od.c*od.h*od.w+1, od.n, _handle);
        _drv->init(DMatrix<T>::One);
        _drv->setT();
        _delta= new DMatrix<T>(od.c*od.h*od.w+1, od.n, _handle);
        _delta->init(DMatrix<T>::One);
        _delta->setT();
    }

    DDim4 output_dim() { return _output_dim; }
    virtual DMatrix<T> *act() { return _act; }
    virtual DMatrix<T> *delta() { return _delta; }
    virtual DNeuron<T> *neuron() { return _neuron; }

    virtual void clearMomentum() {
        _mom_bias->init(DMatrix<T>::Zero);
        _mom_filter->init(DMatrix<T>::Zero);
    }

    virtual void regParams(std::vector<DMatrix<T>*> &X, std::vector<DMatrix<T>*> &dX) {
        X.push_back(_bias);
        X.push_back(_filter);
        dX.push_back(_mom_bias);
        dX.push_back(_mom_filter);
        _neuron->regParams(X, dX);
    }

    virtual void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) {
        const T alpha = 1.0; 
        CUDNN_CALL(cudnnConvolutionForward(_cudnn_handle, _input_desc, dev_data->dev_data(), _filter_desc, _filter->dev_data(), _conv_desc, _output_desc, _drv->dev_data(), CUDNN_RESULT_NO_ACCUMULATE));
        CUDNN_CALL(cudnnAddTensor4d(_cudnn_handle, CUDNN_ADD_SAME_C, &alpha, _bias_desc, _bias->dev_data(), _output_desc, _drv->dev_data()));
        _neuron->fprop(_act, _drv);
    }

    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate, bool rectify_weight, bool rectify_bias) {
        _neuron->bprop(_delta, _drv, _act);        
        if (delta) 
            CUDNN_CALL(cudnnConvolutionBackwardData(_cudnn_handle, _filter_desc, _filter->dev_data(), _output_desc, _delta->dev_data(), _conv_desc, _input_desc, delta->dev_data(), CUDNN_RESULT_NO_ACCUMULATE));
        CUDNN_CALL(cudnnConvolutionBackwardBias(_cudnn_handle, _output_desc, _delta->dev_data(), _bias_desc, _grad_bias->dev_data(), CUDNN_RESULT_NO_ACCUMULATE));
        _mom_bias->applyBinary(OpMom(mom, -(1.0-mom)*rate/_delta->nrows()), _grad_bias, _grad_bias->nrows(), _grad_bias->ncols());
        _bias->add(_mom_bias, 1.0);
        CUDNN_CALL(cudnnConvolutionBackwardFilter(_cudnn_handle, _input_desc, pre_act->dev_data(), _output_desc, _delta->dev_data(), _conv_desc, _filter_desc, _grad_filter->dev_data(), CUDNN_RESULT_NO_ACCUMULATE));
        _mom_filter->applyBinary(OpMom(mom, -(1.0-mom)*rate/_delta->nrows()), _grad_filter, _grad_filter->nrows(), _grad_filter->ncols() - 1);
        _filter->add(_mom_filter, 1.0);
    }
};

#endif //DCONVLAYER_CUH
