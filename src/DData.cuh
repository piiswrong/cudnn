#ifndef DDATA_CUH
#define DDATA_CUH

#include <common.cuh>
#include <algorithm>
#include <DMatrix.cuh>
#include <pthread.h>
#include <cstring>

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
    int _buff_index;
    int _buff_offset;
    pthread_t _thread;
    pthread_cond_t _cond;
    pthread_mutex_t _mutex;

    void moveOn() {
        pthread_mutex_lock[&_mutex];
        _ready[_index++] = false;
        pthread_cond_signal(&_cond);
        pthread_mutex_unlock(&_mutex);
    }
public:
    enum Split {
        Train = 1,
        Validate = 2,
        Test = 4
    };
    DData(int num_buffs, int x_dim, y_dim, int buff_dim, bool permute, cublasStatus_t handle = 0) {
        _num_buffs = num_buffs;
        _permute = permute;
        _x_dim = x_dim;
        _y_dim = y_dim;
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
    
    ~DData() {
        pthread_cancel(_thread);
        if (_on_device) {
            del _perm_seq;
        }
        del _ready;
        for (int i = 0; i < _num_buffs; i++) {
            del _x_buffs[i];
            del _y_buffs[i];
        }
        del _x_buffs;
        del _y_buffs;
        pthread_mutex_destroy(&_mutex);
        pthread_cond_destroy(&_cond);
    }

    int x_dim() { return _x_dim; }
    int y_dim() { return _y_dim; }

    void start() {
        pthread_create(&_thread, NULL, DData<T>::generateDataHelper, (void*)this);
    }

    static void *generateDataHelper(void *this) {
        ((DData<T>*)this)->generateData();
    }

    void generateData() {
        Tx *x;
        Ty *y;
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
            del x;
            del y;
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

    virtual bool fetch(T *&x, T *&y) = 0;
    virtual int instancesPerEpoch() = 0;
    
    bool getData(DMatrix<T> *&x, DMatrix<T> *&y, int batch_size) {
        if (batch_size > _buff_dim) return false;

        pthread_mutex_lock(&_mutex);
        while (!_ready[_index]) pthread_cond_wait(&_cond, &_mutex);
        pthread_mutex_unlock(&_mutex);
        
        if (batch_size > _buff_dim - _buff_offset) {
            moveOn();
            return getData(x, y, batch_size);
        }else {
            x = new DMatrix<T>(_x_buffs[_index], _buff_offset, batch_size);
            y = new DMatrix<T>(_y_buffs[_index], _buff_offset, batch_size);
            _buff_offset += batch_size;
            return true;
        }
    }


};

template<class T>
class DMnistData : DData<T> {
    int _split;
    FILE *_xfile;
    FILE *_yfile;
    int _offset;
    int _soffset;
    int _eoffset;

    char *_tx, *_ty;
public:
    DMnistData(string path, int split, cublasStatus_t handle = 0) : DData(2, 28*28, 10, 10000, true, handle) {
        _split = split;
		if (path[path.length()-1] != '/') path.append("/");
        if (split&(Split::Train|Split::Validate)) {
            _xfile = fopen(strFile+"train-images-idx3-ubyte", "r");
			_yfile = fopen(strFile+"train-labels-idx1-ubyte", "r");
			_soffset = 0;
			_eoffset = 60000;
			if (!(split&Split::Train)) _soffset = 50000;
			if (!(split&Split::Validate)) _eoffset = 50000;
        }else {
			_xfile = fopen(strFile+"t10k-images-idx3-ubyte", "r");
			fseek(_xfile, 16, SEEK_SET);
			_yfile = fopen(strFile+"t10k-labels-idx1-ubyte", "r");
			fseek(_yfile, 8, SEEK_SET);
			_soffset = 0;
			_eoffset = 10000;
		}
		_offset = 0;
		fseek(_xfile, 16+_soffset*_x_dim, SEEK_SET);
		fseek(_yfile, 8+_soffset, SEEK_SET);

        _tx = new char[_x_dim*_buff_dim];
        _ty = new char[_buff_dim];
    }
	virtual int instancesPerEpoch () { return _eoffset - _soffset; }
	virtual bool fetch(T *&x ,T *&y) {
		x = new T[_x_dim*_buff_dim];
        y = new T[_y_dim*_buff_dim];
        int need = _buff_dim;
        do {
            int available = (need < _eoffset - _soffset) ? need:(_eoffset - _soffset);
            fread(_tx, sizeof(char), available*_x_dim, _xfile);
            fread(_ty, sizeof(char), available, _yfile);
            need -= available;
            _offset += available;
            if (_offset == _eoffset) {
		        fseek(_xfile, 16, SEEK_SET);
		        fseek(_yfile, 8, SEEK_SET);
                _offset = 0;
            }
        }while (need);

        for (int i = 0; i < _buff_dim*_x_dim; i++) x[i] = (T)_tx[i]/256.0;
        memset(y, 0, sizeof(T)*_y_dim*_buff_dim);
        for (int i = 0; i < _buff_dim; i++) y[i*_y_dim + _ty[i]] = 1.0;
        return true;
	}

};

#endif //DDATA_CUH
