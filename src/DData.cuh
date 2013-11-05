#ifndef DDATA_CUH
#define DDATA_CUH

#include <common.cuh>
#include <algorithm>
#include <DMatrix.cuh>
#include <pthread.h>

template<class T>
class DData {
    bool _on_device;
    cublasStatus_t _handle;
    bool _permute;
    int *_perm_seq;
    int _x_dim;
    int _y_dim;
    int _buff_dim;
    DMatrix<T> **_x_buffs;
    DMatrix<T> **_y_buffs;
    volatile bool *_ready;
    int _index;
    pthread_cond_t _cond;
    pthread_mutex_t _mutex;

public:
    DData(int num_buffs, int data_dim, int buff_dim, bool permute, cublasStatus_t handle = 0) {
        _num_buffs = num_buffs;
        _permute = permute;
        _data_dim = data_dim;
        _buff_dim = buff_dim;
        _handle = handle;
        if (_handle) {
            _on_device = true;
        }

        if (_permute) {
            _perm_seq = new int[_buff_dim];
            for (int i = 0; i < _buff_dim; i++) _perm_seq[i] = i;
        }

        _index = 0;
        _ready = new bool[_num_buffs];
        _x_buffs = new DMatrix<T>*[_num_buffs];
        _y_buffs = new DMatrix<T>*[_num_buffs];
        for (int i = 0; i < _num_buffs; i++) {
            _x_buffs[i] = new DMatrix<T>(_x_dim, _buff_dim, _handle);
            _x_buffs[i]->setT();
            _y_buffs[i] = new DMatrix<T>(_y_dim, _buff_dim, _handle);
            _y_buffs[i]->setT();
            _ready[i] = false;
        }
        pthread_mutex_init(&_mutex);
        pthread_cond_init(&_cond);
    }
    
    int x_dim() { return _x_dim; }
    int y_dim() { return _y_dim; }

    static void *generateDataHelper(void *this) {
        ((DData<T>*)this)->generateData();
    }

    void generateData() {
        T *x, *y;
        for (int c = 0; ; c = (c+1)%_num_buffs) {
            pthread_mutex_lock(&_mutex);
            while (_ready[c]) {
                pthread_cond_wait(&_cond, &_mutex);
            }
            pthread_mutex_unlock(&_mutex);
            fetch(x, y);
            if (_permute) {
                for (int i = 0; i < _buff_dim; i++) {
                    j = rand()%(_buff_dim - i);
                    swap(_perm_seq[i], _perm_seq[j+i]);
                }
                for (int i = 0; i < _buff_dim; i++) {
                    memcpy(x+_x_dim*i, _x_buffs[c]->host_data() + _x_dim*_perm_seq[i], sizeof(T)*_x_dim);
                    memcpy(y+_y_dim*i, _y_buffs[c]->host_data() + _y_dim*_perm_seq[i], sizeof(T)*_y_dim);
                }
            }else {
                memcpy(x, _x_buffs[c]->host_data(), _x_buffs[c]->size()); 
                memcpy(y, _y_buffs[c]->host_data(), _y_buffs[c]->size());
            }
            if (_on_device) {
                _x_buffs[c]->host2dev();
                _y_buffs[c]->host2dev();
            }
            pthread_mutex_lock(&_mutex);
            _ready[c] = true;
            pthread_cond_signal(&_cond);
            pthread_mutex_unlock(&_mutex);
        }
    }

    virtual void fetch(T *&x, T *&y) = 0;
    
    void getData(DMatrix<T> *&x, DMatrix<T> *&y) {
        pthread_mutex_lock(&_mutex);
        while (!_ready[_index]) pthread_cond_wait(&_cond, &_mutex);
        pthread_mutex_unlock(&_mutex);
        x = _x_buffs[_index];
        y = _y_buffs[_index];
    }

    void moveOn() {
        pthread_mutex_lock[&_mutex];
        _ready[_index++] = false;
        pthread_cond_signal(&_cond);
        pthread_mutex_unlock(&_mutex);
    }

};



#endif //DDATA_CUH
