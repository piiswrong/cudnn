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
        initDelta(delta, act, y);
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
            _loss = delta->dot(delta, delta->nelem() - delta->ld());
        }
        _loss /= 2.0;
    }
    virtual T getLoss() {
        return _loss;
    }
    virtual T objective(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        initDelta(delta, act, y);
        computeLoss(delta, act, y);
        return getLoss()/y->nrows();
    }
    virtual int params(DMatrix<T> **&X, DMatrix<T> **&dX, int *&M, int *&N) {
        return 0;
    }

    virtual void samplePrint() {
    }

    virtual void test_output(std::ofstream &fout, DMatrix<T> *x, DMatrix<T> *y, DMatrix<T> *act) {
        act->dev2host();
        for (int i = 0; i < y->nrows(); i++) {
            for (int j = 0; j < act->ncols()-1; j++) {
                fout << act->getElem(i,j);
                if (j == act->ncols()-2) fout << "\n";
                else fout << " ";
            }
        }
    }
};

template<class T>
class DCutoffNeuron : public DNeuron<T> {
public:
    class ForwardOp {
    public:
        HOSTDEVICE T operator() (T act, T drv) {
            return min(drv, 1.0);
        }
    };
    class BackwardOp {
    public:
        HOSTDEVICE T operator() (T delta, T drv, T act) {
            return delta*(act < (T)1.0);
        }
    };

    DCutoffNeuron(cublasHandle_t handle) : DNeuron<T>(handle) {}
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
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
class DTanhNeuron : public DNeuron<T> {
public:
    class ForwardOp {
    public:
        HOSTDEVICE T operator() (T act, T drv) {
            T e = exp(-2.0*drv);
            return (1.0-e)/(1.0+e);
        }
    };
    class BackwardOp {
    public:
        HOSTDEVICE T operator() (T delta, T drv, T act) {
            return delta*(1.0-act*act);
        }
    };

    DTanhNeuron(cublasHandle_t handle) : DNeuron<T>(handle) {}
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
    DMatrix<T> *obj;
public:
    DSoftmaxNeuron(int batch_size, cublasHandle_t handle) : DNeuron<T>(handle) {
        res = new DMatrix<int>(batch_size, 1, handle);
        obj = NULL;
    }
    virtual bool easyDropout() { return false; }
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
#ifndef DISABLE_GPU
        if (DNeuron<T>::_on_device) {
            dim3 grid((act->ld()-1)/WARP_SIZE+1, 1, 1);
            dim3 block(WARP_SIZE, 32, 1);
            kSoftmaxAct<T,32><<<grid, block>>>(act->dev_data(), drv->dev_data(), res->dev_data(), act->ld(), act->fd(), act->fd()-1);
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
    virtual T objective(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        if (obj == NULL) obj = new DMatrix<T>(delta->nrows(), delta->ncols()-1, delta->handle());
        obj->applyTenary(OpWeightedLog<T>(), act, y, obj->nrows(), obj->ncols());
        return obj->norm1(obj->nelem())/y->nrows();
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
    DHyperParams *_hyper_params;
    T _lambda, _margin;
    DMatrix<T> *_centers, *_dist, *_mask;
    DMatrix<T> *_min_dist, *_coef;

    DMatrix<int> *_ind;

    T _loss;

    
    
public:
    class OpDelta {
        T _m, _lambda;
    public:
        OpDelta(T m, T lambda) : _m(m), _lambda(lambda) {}
        HOSTDEVICE T operator() (T y, T d) {
            T t = _m - d;
            t = t*(t>0);
            return _lambda*y - (1.0-_lambda)*(1.0-y)*t/(d+0.000001);
        }
    };

    DClusterNeuron(DHyperParams *hyper_params, int n_centers, int n_dims, T lambda, T margin, cublasHandle_t handle) : DNeuron<T>(handle) {
        int batch_size = hyper_params->batch_size;
        _lambda = lambda;
        _margin = margin;
        _hyper_params = hyper_params;

        _centers = new DMatrix<T>(n_dims, n_centers, handle);
        _centers->init(DMatrix<T>::Normal, 0.0, 1.0);
        _dist = new DMatrix<T>(batch_size, n_centers, handle);
        _mask = new DMatrix<T>(batch_size, n_centers, handle);
        _min_dist = new DMatrix<T>(batch_size, 1, handle);
        _coef = new DMatrix<T>(batch_size, 1, handle);
        _ind = new DMatrix<int>(batch_size, 1, handle);
    }

    virtual void fprop(DMatrix<T> *act, DMatrix<T> *drv) {
        act->CopyFrom(drv);
        DMatrix<T> *drv_view = new DMatrix<T>(drv, 0, drv->fd()-1);
        DMatrix<T> *act_view = new DMatrix<T>(act, 0, act->fd()-1);
        hComputeDistanceKernel(DistEuclid<T>(), act_view, _centers, _dist);
        _dist->applyBinary(OpSqrt<T>(), _dist, _dist->nrows(), _dist->ncols());
        //_dist->samplePrint("dist");
        hReduce(OpMinReduce<T>(), _dist, _ind, _min_dist, _dist->ncols());
        //_min_dist->samplePrint("min_dist");
        //_ind->samplePrint("ind");
    }

    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _coef->CopyFrom(y);
        _coef->setT();
    }

    virtual void bprop(DMatrix<T> *delta, DMatrix<T> *drv, DMatrix<T> *act) {
        DMatrix<T> *delta_view = new DMatrix<T>(delta, 0, delta->fd()-1);
        DMatrix<T> *drv_view = new DMatrix<T>(drv, 0, drv->fd()-1);
        DMatrix<T> *act_view = new DMatrix<T>(act, 0, act->fd()-1);
        delta_view->CopyFrom(act_view);
        hDecode(_mask, _ind);
        delta_view->update(_mask, false, _centers, true, -1.0, 1.0);
        //delta->samplePrint("delta1");
        //_coef->samplePrint("y");
        _coef->applyBinary(OpDelta(_margin, _lambda), _min_dist, _coef->nrows(), _coef->ncols());
        //_coef->samplePrint("coef");
        delta_view->diagMul(delta_view, _coef, true);
        //delta->samplePrint("delta2");
    }

    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        y->dev2host();
        _min_dist->dev2host();
        _loss = 0.0;
        for(int i = 0; i < y->nrows(); i++) {
            T yi = y->getElem(i,0);
            T di = _min_dist->getElem(i,0);
            T ndi = max(0.0, _margin-di);
            _loss += _lambda*yi*di*di*0.5 + (1.0-_lambda)*(1.0-yi)*ndi*ndi*0.5;
        }
    }

    virtual T getLoss() {
        return _loss;
    }

    /*virtual int params(DMatrix<T> **&X, DMatrix<T> **&dX, int *&M, int *&N) {
        int L = 1;
        X = new DMatrix<T>*[L];
        dX = new DMatrix<T>*[L];
        M = new int[L];
        N = new int[L];
        X[0] = _means;
        dX[0] = _mom_means;
        M[0] = _means->nrows();
        N[0] = _means->ncols();
        return L;
    }*/

    virtual void samplePrint() {
        _centers->samplePrint("centers");
        _dist->samplePrint("dist");
    }

    virtual void test_output(std::ofstream &fout, DMatrix<T> *x, DMatrix<T> *y, DMatrix<T> *act) {
        _ind->dev2host();
        for (int i = 0; i < y->nrows(); i++) {
            fout << _ind->getElem(i,0) << "\n";
        }
    }

};

template<class T>
class DvMFNeuron : public DNeuron<T> {
    DHyperParams *_hyper_params;
    T _lambda;
    DMatrix<T> *_means, *_kappa, *_gamma, *_pi, *_coef, *_dist, *_likelyhood, *_norm;
    DMatrix<T> *_mom_means, *_mom_kappa, *_mom_pi; 
    DMatrix<T> *_dmeans;
    DMatrix<T> *_tmpk, *_tmpn, *_tmpnl, *_max_dist, *_max_kappa;
    T _loss;
public:
    DvMFNeuron(DHyperParams *hyper_params, int n_centers, int n_dims, T lambda, cublasHandle_t handle) : DNeuron<T>(handle) {
        int batch_size = hyper_params->batch_size;
        _lambda = lambda;
        _hyper_params = hyper_params;

        _tmpk = new DMatrix<T>(n_centers, 1, handle);
        _tmpn = new DMatrix<T>(batch_size, 1, handle);
        _tmpnl = new DMatrix<T>(batch_size, n_dims, handle);
        _means = new DMatrix<T>(n_dims, n_centers, handle);
        _means->init(DMatrix<T>::Normal, 0.0, 1.0);
        hNormalize<T, OpSqr<T>, OpSumReduce<T>, OpSqrt<T>, OpDivide<T> >(OpSqr<T>(), OpSumReduce<T>(), OpSqrt<T>(), OpDivide<T>(), _means, _means, NULL, _means->nrows(), true);
        _kappa = new DMatrix<T>(n_centers, 1, handle);
        _kappa->init(DMatrix<T>::Uniform, 10.0, 10.0);
        _mom_means = new DMatrix<T>(n_dims, n_centers, handle);
        _mom_means->init(DMatrix<T>::Zero);
        _dmeans = new DMatrix<T>(n_dims, n_centers, handle);
        _mom_kappa = new DMatrix<T>(n_centers, 1, handle);
        _mom_kappa->init(DMatrix<T>::Zero);
        _norm = new DMatrix<T>(batch_size, 1, handle);
        _coef = new DMatrix<T>(batch_size, 1, handle);
        _dist = new DMatrix<T>(batch_size, n_centers, handle);
        _gamma = new DMatrix<T>(batch_size, n_centers, handle);
        _pi = new DMatrix<T>(n_centers, 1, handle);
        _pi->init(DMatrix<T>::Uniform, 1.0/n_centers, 1.0/n_centers); 
        _likelyhood = new DMatrix<T>(batch_size, 1, handle);
        _max_dist = new DMatrix<T>(batch_size, 1, handle);
        _max_kappa = new DMatrix<T>(1, 1, handle);
    }

    virtual void fprop(DMatrix<T> *act, DMatrix<T> *drv) { 
        DMatrix<T> *drv_view = new DMatrix<T>(drv, 0, drv->fd()-1);
        DMatrix<T> *act_view = new DMatrix<T>(act, 0, act->fd()-1);
        hNormalize<T, OpSqr<T>, OpSumReduce<T>, OpSqrt<T>, OpDivide<T> >(OpSqr<T>(), OpSumReduce<T>(), OpSqrt<T>(), OpDivide<T>(), drv_view, act_view, _norm, drv_view->fd(), false);
        _dist->update(act_view, false, _means, false, 1.0, 0.0);
        _gamma->diagMul(_dist, _kappa, false);
        hNormalize<T, OpNop<T>, OpMaxReduce<T>, OpNop<T>, OpSub<T> >(OpNop<T>(), OpMaxReduce<T>(), OpNop<T>(), OpSub<T>(), _gamma, _gamma, _max_dist, _gamma->fd(), false);
        _gamma->applyBinary(OpExp<T>(), _gamma, _gamma->nrows(), _gamma->ncols());
        hNormalize<T, OpNop<T>, OpSumReduce<T>, OpNop<T>, OpDivide<T> >(OpNop<T>(), OpSumReduce<T>(), OpNop<T>(), OpDivide<T>(), _gamma, _gamma, _likelyhood, _gamma->fd(), false);
    }

    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _coef->applyBinary(OpGMMDelta<T>(_lambda), y, y->nrows(), y->ncols());
    }

    virtual void bprop(DMatrix<T> *delta, DMatrix<T> *drv, DMatrix<T> *act) {
        DMatrix<T> *drv_view = new DMatrix<T>(drv, 0, drv->fd()-1);
        DMatrix<T> *act_view= new DMatrix<T>(act, 0, act->fd()-1);
        DMatrix<T> *delta_view = new DMatrix<T>(delta, 0, delta->fd()-1);
        _gamma->diagMul(_gamma, _coef, true);
        _gamma->diagMul(_gamma, _kappa, false);
        //dX
        delta_view->update(_gamma, false, _means, true);
        _tmpnl->applyTenary(OpMul<T>(), delta_view, drv_view, delta_view->nrows(), delta_view->ncols());
        hNormalize<T, OpNop<T>, OpSumReduce<T>, OpNop<T>, OpNop<T> >(OpNop<T>(), OpSumReduce<T>(), OpNop<T>(), OpNop<T>(), _tmpnl, _tmpnl, _tmpn, _tmpnl->fd(), false);
        _tmpnl->diagMul(drv, _tmpn, true);
        _tmpn->applyBinary(OpSqr<T>(), _norm, _tmpn->nrows(), _tmpn->ncols());
        delta_view->diagMul(delta_view, _tmpn, true);
        delta_view->applyTenary(OpSub<T>(), delta_view, _tmpnl, delta_view->nrows(), delta_view->ncols());
        _tmpn->applyBinary(OpInvCube<T>(), _norm, _tmpn->nrows(), _tmpn->ncols());
        delta_view->diagMul(delta_view, _tmpn, true);

        //dmu
        _mom_means->update(act_view, true, _gamma, false, -_hyper_params->learning_rate/act->nrows(), _hyper_params->momentum);
        _means->add(_mom_means, 0.0, _means->nelem());
        hNormalize<T, OpSqr<T>, OpSumReduce<T>, OpSqrt<T>, OpDivide<T> >(OpSqr<T>(), OpSumReduce<T>(), OpSqrt<T>(), OpDivide<T>(), _means, _means, NULL, _means->nrows(), true);
    }

    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _likelyhood->dev2host();
        _max_dist->dev2host();
        _loss = 0.0;
        for (int i = 0; i < y->nrows(); i++) {
            T coef = y->getElem(i, 0);
            coef = _lambda*(1.0-coef) - coef;
            _loss += coef * (log(_likelyhood->getElem(i, 0)) + _max_dist->getElem(i, 0));
        }
        _loss = _loss;
        if (_loss != _loss) { 
            printf("Error: nan encontered during training!\n");
            exit(-1);
        }
    }
   
    virtual T getLoss() {
        return _loss;
    }

    virtual int params(DMatrix<T> **&X, DMatrix<T> **&dX, int *&M, int *&N) {
        int L = 1;
        X = new DMatrix<T>*[L];
        dX = new DMatrix<T>*[L];
        M = new int[L];
        N = new int[L];
        X[0] = _means;
        dX[0] = _mom_means;
        M[0] = _means->nrows();
        N[0] = _means->ncols();
        return L;
    }

    virtual void init(DData<T> *data) {
        data->stop();
        data->start();
        DMatrix<T> *x;
        DMatrix<T> *y;
        int i = 0;
        while (i < _means->ncols()) {
            data->getData(x, y, 1);
            if (y->getElem(0,0)) {
                for (int j = 0; j < _means->nrows(); j++) _means->getElem(j, i) = x->getElem(0, j);
                i++;
            }
        }
        data->stop();
        hNormalize<T, OpSqr<T>, OpSumReduce<T>, OpSqrt<T>, OpDivide<T> >(OpSqr<T>(), OpSumReduce<T>(), OpSqrt<T>(), OpDivide<T>(), _means, _means, NULL, _means->nrows(), true);
        _means->samplePrint("initial means");
    }

    virtual void samplePrint() {
        _means->samplePrint("means");
        _dist->samplePrint("dist");
        _gamma->samplePrint("gamma");
    }
};



template<class T>
class DGMMNeuron : public DNeuron<T> {
    DHyperParams *_hyper_params;
    T _lambda;
    DMatrix<T> *_means, *_stds, *_gamma, *_pi, *_coef, *_dist, *_likelyhood;
    DMatrix<T> *_mom_means, *_mom_stds, *_mom_pi; 
    DMatrix<T> *_dmeans;
    DMatrix<T> *_tmpk, *_tmpn, *_min_dist, *_max_std;
    T _loss;
public:
    DGMMNeuron(DHyperParams *hyper_params, int n_centers, int n_dims, T lambda, cublasHandle_t handle) : DNeuron<T>(handle) {
        int batch_size = hyper_params->batch_size;
        _lambda = lambda;
        _hyper_params = hyper_params;

        _means = new DMatrix<T>(n_dims, n_centers, handle);
        _means->init(DMatrix<T>::Normal, 0.0, 1.0);
        _stds = new DMatrix<T>(n_centers, 1, handle);
        _stds->init(DMatrix<T>::Uniform, 1.0, 1.0);
        _mom_means = new DMatrix<T>(n_dims, n_centers, handle);
        _mom_means->init(DMatrix<T>::Zero);
        _dmeans = new DMatrix<T>(n_dims, n_centers, handle);
        _mom_stds = new DMatrix<T>(n_centers, 1, handle);
        _mom_stds->init(DMatrix<T>::Zero);
        _coef = new DMatrix<T>(batch_size, 1, handle);
        _dist = new DMatrix<T>(batch_size, n_centers, handle);
        _gamma = new DMatrix<T>(batch_size, n_centers, handle);
        _pi = new DMatrix<T>(n_centers, 1, handle);
        _pi->init(DMatrix<T>::Uniform, 1.0/n_centers, 1.0/n_centers); 
        _likelyhood = new DMatrix<T>(batch_size, 1, handle);
        _tmpk = new DMatrix<T>(n_centers, 1, handle);
        _tmpn = new DMatrix<T>(batch_size, 1, handle);
        _min_dist = new DMatrix<T>(batch_size, 1, handle);
        _max_std = new DMatrix<T>(1, 1, handle);
    }

    virtual void fprop(DMatrix<T> *act, DMatrix<T> *drv) { 
        act->CopyFrom(drv);
        hComputeDistanceKernel<T, DistEuclid<T> >(DistEuclid<T>(), drv, _means, _dist, drv->fd()-1);
        _gamma->diagMul(_dist, _stds, false);
        hNormalize<T, OpNop<T>, OpMinReduce<T>, OpNop<T>, OpSub<T> >(OpNop<T>(), OpMinReduce<T>(), OpNop<T>(), OpSub<T>(), _gamma, _gamma, _min_dist, _gamma->fd(), false);
        _gamma->applyBinary(OpGaussian<T>(), _gamma, _gamma->nrows(), _gamma->ncols());
        hNormalize<T, OpNop<T>, OpMaxReduce<T>, OpNop<T>, OpDivide<T> >(OpNop<T>(), OpMaxReduce<T>(), OpNop<T>(), OpDivide<T>(), _stds, _tmpk, _max_std, _stds->nrows(), true);
        _tmpk->applyTenary(OpGMMWeight<T>(drv->ncols()-1), _pi, _tmpk, _tmpk->nrows(), _tmpk->ncols()); 
        _gamma->diagMul(_gamma, _tmpk, false);
        hNormalize<T, OpNop<T>, OpSumReduce<T>, OpNop<T>, OpDivide<T> >(OpNop<T>(), OpSumReduce<T>(), OpNop<T>(), OpDivide<T>(), _gamma, _gamma, _likelyhood, _gamma->fd(), false);
    }

    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _coef->applyBinary(OpGMMDelta<T>(_lambda), y, y->nrows(), y->ncols());
    }

    virtual void bprop(DMatrix<T> *delta, DMatrix<T> *drv, DMatrix<T> *act) {
        DMatrix<T> *drv_view = new DMatrix<T>(drv, 0, drv->fd()-1);
        DMatrix<T> *delta_view = new DMatrix<T>(delta, 0, delta->fd()-1);
        _gamma->diagMul(_gamma, _coef, true);
        //dX
        _tmpn->update(_gamma, false, _stds, false, 1.0, 0.0);
        delta_view->diagMul(drv_view, _tmpn, true);
        _gamma->diagMul(_gamma, _stds, false);
        delta_view->update(_gamma, false, _means, true, 1.0, -1.0);

        //delta->add(act, 1.0, delta->nelem() - delta->ld());

        T rate = -(1.0-_hyper_params->momentum)*_hyper_params->learning_rate/delta->nrows();
        //dmeans
        for (int i = 0; i < _tmpn->nrows(); i++) _tmpn->getElem(i, 0) = 1.0;
        _tmpn->host2dev();
        _tmpk->update(_gamma, true, _tmpn, false, 1.0, 0.0);
        _dmeans->diagMul(_means, _tmpk, false);
        _dmeans->update(drv_view, true, _gamma, false, rate, -rate);
        _mom_means->applyBinary(OpDPSGDMom<T>(_hyper_params->momentum), _dmeans, _mom_means->nrows(), _mom_means->ncols());
        _means->add(_mom_means, 1.0);

        //dstd
    }

    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        //_coef->applyBinary(OpGMMDelta<T>(_lambda), y, y->nrows(), y->ncols());
        //_likelyhood->applyBinary(OpLog<T>(), _likelyhood, _likelyhood->nrows(), _likelyhood->ncols());
        //_coef->dev2host();
        _likelyhood->dev2host();
        _max_std->dev2host();
        _min_dist->dev2host();
        _loss = 0.0;
        for (int i = 0; i < y->nrows(); i++) {
            T coef = y->getElem(i, 0);
            coef = _lambda*(1.0-coef) - coef;
            _loss += coef * (log(_likelyhood->getElem(i, 0)) - 0.5*_min_dist->getElem(i, 0));
        }
        _loss = _loss + y->nrows() * (delta->ncols()-1)/2.0*log(_max_std->getElem(0,0));
        if (_loss != _loss) { exit(-1);}
    }
   
    virtual T getLoss() {
        return _loss;
    }

    virtual int params(DMatrix<T> **&X, DMatrix<T> **&dX, int *&M, int *&N) {
        int L = 1;
        X = new DMatrix<T>*[L];
        dX = new DMatrix<T>*[L];
        M = new int[L];
        N = new int[L];

        X[0] = _means;
        dX[0] = _mom_means;
        M[0] = _means->nrows();
        N[0] = _means->ncols();
        return L;
    }

    virtual void samplePrint() {
        _means->samplePrint("means");
        _dist->samplePrint("dist");
        _gamma->samplePrint("gamma");
        
    }
};

#endif //DNEURON_CUH
