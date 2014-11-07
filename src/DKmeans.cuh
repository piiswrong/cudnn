#ifndef DKMEANS_CUH
#define DKMEANS_CUH

#include <common.cuh>
#include <DNN.cuh>
#include <DData.cuh>
#include <DOperators.cuh>
#include <DNeuron.cuh>

#include <vector>
#include <map>
#include <algorithm>

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

    std::map<int,int> sample(int N, int s) {
        std::vector<int> p;
        std::map<int,int> res;
        for (int i = 0; i < N; i++) p.push_back(i);
        std::random_shuffle(p.begin(), p.end());
        for (int i = 0; i < s; i++) res.insert(std::pair<int,int>(p[i], i));
        return res;
    }

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

    void cluster(int max_iter = 10, T epsilon = 0.05) {
        DMatrix<T> *samples = new DMatrix<T>(_ndim, _ncenter);
        samples->init(DMatrix<T>::Zero, 0.0, 0.0);
        int npos = _data->instancesPerEpoch()/2;
        T prev_loss = 1e20;
        for (int iter = 0; iter < max_iter; iter++) {
            _tmp_centers->init(DMatrix<T>::Zero, 0.0, 0.0);
            _count->init(DMatrix<T>::Zero, 0.0, 0.0);
            _data->stop();
            _data->set_testing(true);
            _data->start();
            std::map<int,int> sample_ind = sample(npos, _ncenter);
            DMatrix<T> *x, *y;
            int i = 0;
            T loss = 0;
            while (_data->getData(x, y, _batch_size)) {
                DMatrix<T> *act = _dnn->testAct(x);

                act->dev2host();
                for (int j = 0; j < x->nrows(); j++) {
                    if (y->getElem(j, 0) == 1) {
                        std::map<int,int>::const_iterator got = sample_ind.find(i);
                        if (got != sample_ind.end()) {
                            int k = got->second;
                            for (int c = 0; c < _ndim; c++) samples->getElem(c, k) = act->getElem(j, c);
                        }
                        i++;
                    }
                }
                
                _min_dist->dev2host();
                for (int j = 0; j < x->nrows(); j++) loss += y->getElem(j,0)*_min_dist->getElem(j,0);

                _mask->diagMul(_mask, y, true);
                DMatrix<T> *act_view = new DMatrix<T>(act, 0, act->fd()-1);
                if (_mask->nrows() != y->nrows()) {
                    _mask->dev2host();
                    for (int i = y->nrows(); i < _mask->nrows(); i++) 
                        for (int j = 0; j < _mask->ncols(); j++) 
                            _mask->getElem(i,j) = 0.0;
                    _mask->host2dev();
                }
                _tmp_centers->update(act_view, true, _mask, false, 1.0, 1.0);
                hNormalize<T, OpNop<T>, OpSumReduce<T>, OpNop<T>, OpNop<T> >(OpNop<T>(), OpSumReduce<T>(), OpNop<T>(), OpNop<T>(), _mask, NULL, NULL, _tmpk, y->nrows(), true);
                _count->applyTenary(OpAdd<T>(), _count, _tmpk, _count->nrows(), _count->ncols());
                delete x,y;
            }
            npos = i;
            samples->host2dev();
            _count->applyBinary(OpInv(), _count, _count->nrows(), _count->ncols());
            _centers->diagMul(_tmp_centers, _count, false);
            
            _centers->dev2host();
            _count->dev2host();
            for (int i = 0; i < _ncenter; i++) {
                if (_count->getElem(i,0) == 0) {
                    for (int j = 0; j < _ndim; j++) 
                        _centers->getElem(j, i) = samples->getElem(j, i);
                }
            }
            _centers->host2dev();
            _centers->samplePrint("centers");

            loss /= npos;
            printf("LOSS: %f\n", loss);
            if ( (prev_loss - loss)/prev_loss < epsilon ) break;
            prev_loss = loss;
            
        }
    }
};


#endif //DKMEANS_CUH
