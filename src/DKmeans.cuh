#ifndef DKMEANS_CUH
#define DKMEANS_CUH

#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>
#include <DOperators.cuh>
#include <DNeuron.cuh>


template<class T>
class DKmeans {
    cublasHandle_t _handle;
    DNN<T> *_dnn;
    DData<T> *_data;
    int _ndim, _ncenter, _batch_size;
    DMatrix<T> *_centers, *_min_dist, *_mask;

    DMatrix<T> *_tmp_centers, *_count, *_tmpk;

public:
    class OpInv {
    public:
        HOSTDEVICE T operator() (T x, T y) {
            if (y == 0.0) return 0.0;
            else return 1.0/y;
        }
    };

    DKmeans(DNN<T> *dnn, DData<T> *data, int batch_size, DMatrix<T> *centers, DMatrix<T> *mask, DMatrix<T> *min_dist, cublasHandle_t handle) {
        _handle = handle;
        _dnn = dnn;
        _data = data;
        _centers = centers;
        _min_dist = min_dist;
        _mask = mask;
        _ndim = _centers->nrows();
        _ncenter = _centers->ncols();
        _batch_size = batch_size;

        _tmp_centers = new DMatrix<T>(_ndim, _ncenter, _handle);
        _count = new DMatrix<T>(_ncenter, 1, _handle);
        _tmpk = new DMatrix<T>(_ncenter, 1, _handle);
    }

    void cluster(int max_iter) {
        for (int iter = 0; iter < max_iter; iter++) {
            _tmp_centers->init(DMatrix<T>::Zero, 0.0, 0.0);
            _count->init(DMatrix<T>::Zero, 0.0, 0.0);
            _data->stop();
            _data->set_testing(true);
            _data->start();
            DMatrix<T> *x, *y;
            _centers->samplePrint("centers before");
            while (_data->getData(x, y, _batch_size)) {
                printf("%d %d", x->nrows(), x->ncols());
                DMatrix<T> *act = _dnn->testAct(x);
                DMatrix<T> *act_view = new DMatrix<T>(act, 0, act->fd()-1);
                if (_mask->nrows() != y->nrows()) {
                    _mask->dev2host();
                    for (int i = y->nrows(); i < _mask->nrows(); i++) 
                        for (int j = 0; j < _mask->ncols(); j++) 
                            _mask->getElem(i,j) = 0.0;
                }
                _tmp_centers->update(act_view, true, _mask, false, 1.0, 1.0);
                act_view->samplePrint("act");
                _mask->samplePrint("mask");
                //_tmp_centers->samplePrint("center");
                hNormalize<T, OpNop<T>, OpSumReduce<T>, OpNop<T>, OpNop<T> >(OpNop<T>(), OpSumReduce<T>(), OpNop<T>(), OpNop<T>(), _mask, NULL, NULL, _tmpk, y->nrows(), true);
                //_mask->samplePrint("mask");
                //_tmpk->samplePrint("tmp");
                _count->applyTenary(OpAdd<T>(), _count, _tmpk, _count->nrows(), _count->ncols());
                delete x,y;
            }
            _count->samplePrint("count1");
            _count->applyBinary(OpInv(), _count, _count->nrows(), _count->ncols());
            _count->samplePrint("count2");
            _tmp_centers->samplePrint("tmp center");
            _centers->diagMul(_tmp_centers, _count, false);
            _centers->samplePrint("centers after");
        }
    }
};


#endif //DKMEANS_CUH
