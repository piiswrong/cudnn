#ifndef DNEURON_CUH
#define DNEURON_CUH

#include <common.cuh>
#include <DMatrix.cuh>
#include <DOperators.cuh>
#include <kernels.cuh>
#include <stdio.h>

template<class T>
class DNeuron {
protected:
    bool _on_device;
    cublasHandle_t _handle;
    T _loss;
public:
    class ForwardOp {
    public:
        HOSTDEVICE inline T operator() (T act, T drv) {
            return drv;
        }
    };
    class BackwardOp {
    public:
        HOSTDEVICE inline T operator() (T delta, T drv, T act) {
            return delta*1.0;
        }
    };
    class DeltaOp{
    public:
        HOSTDEVICE inline T operator() (T x, T y, T z) {
            return y - z;
        }
    };

    DNeuron(cublasHandle_t handle) {
        _handle = handle;
        if (_handle != 0)
            _on_device = true;
        else
            _on_device = false;
    }
    virtual bool easyDropout() { return true; }
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
    }
    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        delta->applyTenary(DeltaOp(), act, y, y->nrows(), delta->ncols() - 1);
    }
    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        if (delta->nrows() != y->nrows()) {
            _loss = 0.0;
            delta->dev2host();
            for (int i = 0; i < y->ncols(); i++) {
                for (int j = 0; j < y->nrows(); j++) {
                    T t = delta->getElem(j, i);
                    _loss += t*t;
                }
            }
        }else {
            _loss = delta->norm2(delta->nelem() - delta->ld());
        }
    }
    virtual T getLoss() {
        return _loss;
    }
};

template<class T>
class DReLUNeuron : public DNeuron<T> {
public:
    class ForwardOp {
    public:
        HOSTDEVICE T operator() (T act, T drv) {
            return drv*(drv > (T)0.0);
        }
    };
    class BackwardOp {
    public:
        HOSTDEVICE T operator() (T delta, T drv, T act) {
            return delta*(act > (T)0.0);
        }
    };

    DReLUNeuron(cublasHandle_t handle) : DNeuron<T>(handle) {}
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
    }
};

template<class T>
class DLogisticNeuron : public DNeuron<T> {
public:
    class ForwardOp {
    public:
        HOSTDEVICE T operator() (T act, T drv) {
            return 1.0/(1.0+exp(-drv));
        }
    };
    class BackwardOp {
    public:
        HOSTDEVICE T operator() (T delta, T drv, T act) {
            return delta*act*(1.0-act);
        }
    };

    DLogisticNeuron(cublasHandle_t handle) : DNeuron<T>(handle) {}
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
    }
};

template<class T>
class DSoftmaxNeuron : public DNeuron<T> {
protected:
    DMatrix<int> *res;
    DMatrix<T> *_y;
public:
    DSoftmaxNeuron(int batch_size, cublasHandle_t handle) : DNeuron<T>(handle) {
        res = new DMatrix<int>(batch_size, 1, handle);
    }
    virtual bool easyDropout() { return false; }
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
#ifndef DISABLE_GPU
        if (DNeuron<T>::_on_device) {
            dim3 grid((act->ld()-1)/WARP_SIZE+1, 1, 1);
            dim3 block(WARP_SIZE, 32, 1);
            kSoftmaxAct<T,32><<<grid, block>>>(act->dev_data(), drv->dev_data(), res->dev_data(), act->ld(), act->fd()-1);
            CUDA_KERNEL_CHECK();
#ifndef NDEBUG
            act->dev2host();
#endif
        }else 
#endif
        {
            int m = act->ld(), n = act->fd() - 1;
            for (int i = 0; i < m; i++) {
                T myMax = drv->getElem(i,0);
                T myMark = 0;
                for (int j = 1; j < n; j++) {
                    if (drv->getElem(i,j) > myMax) {
                        myMax = drv->getElem(i,j);
                        myMark = j;
                    }
                }
                res->getElem(i,0) = myMark;
                T mySum = 0;
                for (int j = 0; j < n; j++) {
                    mySum += act->getElem(i,j) = exp(drv->getElem(i,j)-myMax);
                }
                for (int j = 0; j < n; j++) {
                    act->getElem(i,j) /= mySum;
                }
            }
        }
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        return;
    }
    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _y = y;
    }
    virtual T getLoss() {
        T loss = 0;
        res->dev2host();
        for (int i = 0; i < _y->nrows(); i++) loss += _y->getElem(i, res->getElem(i, 0));
        return loss;
    }
};

template<class T>
class DOddrootNeuron : public DNeuron<T> {
public:
    class ForwardOp {
    public:
        HOSTDEVICE T operator() (T act, T drv) {
            T x = drv, x0 = 0;
            do {
                x0 = x;
                x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
            }while (abs((x-x0)/x)>1e-6);
            return x;
        }
    };

    class BackwardOp {
    public:
        HOSTDEVICE T operator() (T delta, T drv, T act) {
            return delta/(3.0*act*act+1.0);
        }
    };

    DOddrootNeuron(cublasHandle_t handle) : DNeuron<T>(handle) {}
    virtual bool easyDropout() { return false; }
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
    }

};

/*
template<>
HOSTDEVICE float DOddrootNeuron<float>::ForwardOp::operator() (float act, float drv) {
/*    const unsigned int ebits = 8;
    const unsigned int fbits = 23;
    const unsigned int bias = (1 << (ebits-1))-1;
    const unsigned int mask = (1<<31)-1;

    float x = drv;
    int& i = (int&) x; 
    i = (((i&mask) - (bias << fbits)) / 3 + (bias << fbits))|(i&~mask);

    x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
    x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
    x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
    x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
    x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
    return x;
    float x = drv, x0 = 0;
    do {
        x0 = x;
        x = (2.0/3.0)*x + (drv - (2.0/3.0)*x)/(3.0*x*x + 1);
    }while (fabs((x-x0)/x)>1e-6);
    return x;

}
*/

template<class T>
class DClusterNeuron : public DNeuron<T> {
    int _batch_size;
    int _n_centers;
    int _n_dims;
    T _lambda;
    DMatrix<T> *_centers, *_distance, *_res, *_margin, *_norm, *_scale;
    DMatrix<int> *_index;

    DMatrix<T> *_y, *_acc;


public:
    DClusterNeuron(int batch_size, int n_centers, int n_dims, T lambda, cublasHandle_t handle) : DNeuron<T>(handle) {
        _batch_size = batch_size;
        _n_centers = n_centers;
        _n_dims = n_dims;
        _lambda = lambda;

        _centers = new DMatrix<T>(batch_size, n_dims, DNeuron<T>::_handle);
        _centers->init(DMatrix<T>::Normal);
        UnitLength(_centers, _centers, _norm, n_dims);
        _index = new DMatrix<int>(batch_size, 1, DNeuron<T>::_handle);
        _distance = new DMatrix<T>(batch_size, n_dims, DNeuron<T>::_handle);
        _res = new DMatrix<T>(batch_size, 1, DNeuron<T>::_handle);
        _margin = new DMatrix<T>(n_centers, 1, DNeuron<T>::_handle);
        _norm = new DMatrix<T>(batch_size, 1, DNeuron<T>::_handle);
        _scale = new DMatrix<T>(batch_size, 1, DNeuron<T>::_handle);
        _acc = new DMatrix<T>(batch_size, 1, DNeuron<T>::_handle);
    }
    virtual bool easyDropout() { return false; }

    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        CluterNeuronDelta<T>(_scale, y, _margin, _res, _index, _lambda);
    }

    virtual void fprop(DMatrix<T> *act, DMatrix<T> *drv) {
        drv->samplePrint("drv");
        UnitLength(drv, act, _norm, act->fd()-1);
        act->samplePrint("act");
        DMatrix<T> *act_view = new DMatrix<T>(act, 0, act->fd()-1);
        _distance->update(act_view, false, _centers, true, 1.0, 0);
        _distance->samplePrint("dist");
        Argmax(_distance, _index, _res, _distance->fd());
    }

    virtual void bprop(DMatrix<T> *delta, DMatrix<T> *drv, DMatrix<T> *act) {
        CluterNeuronBprop(delta, act, _centers, _index, _res, _scale, _norm, delta->fd()-1);
        delta->samplePrint("delta");
    }

    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _y = y;
    }

    virtual T getLoss() {
        CluterNeuronAcc(_acc, _y, _margin, _res, _index);
        return _acc->norm1(_y->nrows());
    }
};

#endif //DNEURON_CUH
