#ifndef DDATA_CUH
#define DDATA_CUH

#include <common.cuh>
#include <algorithm>
#include <DMatrix.cuh>
#include <DOperators.cuh>
#include <pthread.h>
#include <cstring>

template<class T>
class DData {
protected:
    bool _on_device;
    cublasHandle_t _handle;
    bool _started;
    bool _permute;
    int *_perm_seq;
    int _x_dim;
    int _y_dim;
    int _num_buffs;
    int _buff_dim;
    DMatrix<T> **_x_buffs;
    DMatrix<T> **_y_buffs;
    volatile bool *_ready;
    volatile int *_available;
    cudaStream_t *_streams;
    int _buff_index;
    int _buff_offset;
    bool _testing;
    pthread_t _thread;
    pthread_cond_t _cond_get;
    pthread_cond_t _cond_gen;
    pthread_mutex_t _mutex;
    
    volatile bool _cancel_flag;

    void moveOn() {
        pthread_mutex_lock(&_mutex);
        _ready[_buff_index]= false;
        _buff_index = (_buff_index + 1)%_num_buffs;
        _buff_offset = 0;
        pthread_mutex_unlock(&_mutex);
        pthread_cond_signal(&_cond_gen);
    }
public:
    enum Split {
        Train = 1,
        Validate = 2,
        Test = 4
    };
    DData(int num_buffs, int x_dim, int y_dim, int buff_dim, bool permute, bool testing, cublasHandle_t handle) {
        _started = false;
        _cancel_flag = false;
        _num_buffs = num_buffs;
        _permute = permute;
        _x_dim = x_dim;
        _y_dim = y_dim;
        _buff_dim = buff_dim;
        _handle = handle;
        _testing = testing;
        if (_handle) {
            _on_device = true;
            _streams = new cudaStream_t[num_buffs];
            for (int i = 0; i < num_buffs; i++) CUDA_CALL(cudaStreamCreate(_streams+i));
        }

        if (_permute) {
            _perm_seq = new int[_buff_dim];
        }

        _ready = new bool[_num_buffs];
        _available = new int[_num_buffs];
        _x_buffs = new DMatrix<T>*[_num_buffs];
        _y_buffs = new DMatrix<T>*[_num_buffs];
        for (int i = 0; i < _num_buffs; i++) {
            _x_buffs[i] = new DMatrix<T>(_x_dim, _buff_dim, _handle);
            _x_buffs[i]->setT();
            _y_buffs[i] = new DMatrix<T>(_y_dim, _buff_dim, _handle);
            _y_buffs[i]->setT();
        }
    }
    
    ~DData() {
        stop();
        if (_on_device) {
            delete _perm_seq;
        }
        delete _ready;
        delete _available;
        for (int i = 0; i < _num_buffs; i++) {
            delete _x_buffs[i];
            delete _y_buffs[i];
        }
        delete _x_buffs;
        delete _y_buffs;
        pthread_mutex_destroy(&_mutex);
        pthread_cond_destroy(&_cond_gen);
        pthread_cond_destroy(&_cond_get);
    }

    int x_dim() { return _x_dim; }
    int y_dim() { return _y_dim; }

    virtual void start() {
        if (!_started) {
            _started = true;
            _buff_index = 0;
            _buff_offset = 0;
            for (int i = 0; i < _num_buffs; i++) {
                _ready[i] = false;
                _available[i] = 0;
            }
            pthread_mutex_init(&_mutex, NULL);
            pthread_cond_init(&_cond_get, NULL);
            pthread_cond_init(&_cond_gen, NULL);
            pthread_create(&_thread, NULL, DData<T>::generateDataHelper, (void*)this);
        }
    }

    virtual void stop() {
        if (_started) {
            _cancel_flag = true;
            pthread_mutex_lock(&_mutex);
            pthread_cond_signal(&_cond_gen);
            pthread_mutex_unlock(&_mutex);
            pthread_join(_thread, NULL);
            _cancel_flag = false;
            _started = false;
        }
    }

    virtual void balance(int s, int t, int sec) {}
    virtual int totalInstancesPerEpoch() { return instancesPerEpoch(); }

    static void *generateDataHelper(void *ddata) {
        ((DData<T>*)ddata)->generateData();
        return NULL;
    }

    void generateData() {
        T *x=0;
        T *y=0;
        for (int c = 0; !_cancel_flag; c = (c+1)%_num_buffs) {
            pthread_mutex_lock(&_mutex);
            while (_ready[c]) {
                if (_cancel_flag) break;
                pthread_cond_wait(&_cond_gen, &_mutex);
            }
            pthread_mutex_unlock(&_mutex);
            if (_cancel_flag) return;
            int n = fetch(x, y);
            if (_permute) {
                for (int i = 0; i < n; i++) 
                    _perm_seq[i] = i;
                /*for (int i = 0; i < n; i++) {
                    int j = rand()%(n - i);
                    std::swap(_perm_seq[i], _perm_seq[j+i]);
                }*/
                for (int i = 0; i < n; i++) {
                    memcpy(_x_buffs[c]->host_data() + _x_dim*_perm_seq[i], x+_x_dim*i, sizeof(T)*_x_dim);
                    memcpy(_y_buffs[c]->host_data() + _y_dim*_perm_seq[i], y+_y_dim*i, sizeof(T)*_y_dim);
                }
            }else {
                memcpy(_x_buffs[c]->host_data(), x, sizeof(T)*n*_x_dim);
                memcpy(_y_buffs[c]->host_data(), y, sizeof(T)*n*_y_dim);
            }
            delete x;
            delete y;
            if (_on_device) {
                _x_buffs[c]->host2devAsync(_streams[c]);
                _y_buffs[c]->host2devAsync(_streams[c]);
            }
            pthread_mutex_lock(&_mutex);
            _ready[c] = true;
            _available[c] = n;
            pthread_cond_signal(&_cond_get);
            pthread_mutex_unlock(&_mutex);
            if (n < _buff_dim) break;
        }
    }

    virtual int fetch(T *&x, T *&y) = 0;
    virtual int instancesPerEpoch() = 0;
    
    virtual bool getData(DMatrix<T> *&x, DMatrix<T> *&y, int batch_size) {
        if (!_testing && batch_size > _buff_dim) return false;

        pthread_mutex_lock(&_mutex);
        while (!_ready[_buff_index]) {
            pthread_cond_wait(&_cond_get, &_mutex);
        }
        pthread_mutex_unlock(&_mutex);
        
        int dim = _available[_buff_index];
        if (batch_size > dim - _buff_offset) {
            if (_testing) {
                x = new DMatrix<T>(_x_buffs[_buff_index], _buff_offset, dim - _buff_offset);
                y = new DMatrix<T>(_y_buffs[_buff_index], _buff_offset, dim - _buff_offset);
                x->setT();
                x->setT();
                CUDA_CALL(cudaStreamSynchronize(_streams[_buff_index]));
                if (dim < _buff_dim) {
                    return false;
                }else {
                    moveOn();
                    return true;
                }
            }else {
                moveOn();
                return getData(x, y, batch_size);
            }
        }else {
            x = new DMatrix<T>(_x_buffs[_buff_index], _buff_offset, batch_size);
            y = new DMatrix<T>(_y_buffs[_buff_index], _buff_offset, batch_size);
            x->setT();
            x->setT();
            CUDA_CALL(cudaStreamSynchronize(_streams[_buff_index]));
            _buff_offset += batch_size;
            return true;
        }
    }


};

template<class T, class Tx, class Ty, class xOp, class yOp>
class DBinaryData : public DData<T> {
protected:
    FILE *_xfile;
    FILE *_yfile;
    int _xskip;
    int _yskip;
    int _offset;
    int _soffset;
    int _eoffset;

    bool _xappendone;
    bool _yonehot;

    int _xdim;
    int _ydim;

    xOp _xop;
    yOp _yop;

    Tx *_tx;
    Ty *_ty;

public:
    DBinaryData(xOp xop, yOp yop, int xdim, int ydim, bool xappendone, bool yonehot, int buff_dim, bool permute, int testing, cublasHandle_t handle) :
                DData<T>(2, xdim + xappendone, ydim, buff_dim, permute, testing, handle), _xop(xop), _yop(yop) {
        _xappendone = xappendone;
        _yonehot = yonehot;
        _xdim = xdim;
        _ydim = yonehot ? 1:ydim;
        _tx = new Tx[_xdim*buff_dim];
        _ty = new Ty[_ydim*buff_dim];
    }

    ~DBinaryData() {
        DData<T>::stop();
        fclose(_xfile);
        fclose(_yfile);
        delete _tx;
        delete _ty;
    }
    
    void open(const char *xpath, const char *ypath, int xskip, int yskip, int soffset, int eoffset) {
        _xskip = xskip;
        _yskip = yskip;
        _xfile = fopen(xpath, "r");
        _yfile = fopen(ypath, "r");
        _soffset = soffset;
        _eoffset = eoffset;
    }

	virtual int instancesPerEpoch () { return _eoffset - _soffset; }

    virtual void start() {
        if (DData<T>::_started) return;
		_offset = _soffset;
		fseek(_xfile, _xskip+_soffset*_xdim, SEEK_SET);
		fseek(_yfile, _yskip+_soffset*_ydim, SEEK_SET);
        DData<T>::start();
    }
    
    virtual int fetch(T *&x ,T *&y) {
		x = new T[DData<T>::_x_dim*DData<T>::_buff_dim];
        y = new T[DData<T>::_y_dim*DData<T>::_buff_dim];
        int need = DData<T>::_buff_dim;
        do {
            int available = (need < _eoffset - _offset) ? need:(_eoffset - _offset);
            int xread = fread(_tx, sizeof(Tx), available*_xdim, _xfile); 
            int yread = fread(_ty, sizeof(Ty), available*_ydim, _yfile); 
            assert(available*_xdim == xread);
            assert(available*_ydim == yread);
            
            need -= available;
            _offset += available;
            if (_offset == _eoffset) {
                if (!DData<T>::_testing) {
                    fseek(_xfile, _xskip+_soffset*_xdim, SEEK_SET);
                    fseek(_yfile, _yskip+_soffset*_ydim, SEEK_SET);
                    _offset = 0;
                }else {
                    break;
                }
            }
        }while (need);
        
        int ld = _xdim, fd = DData<T>::_buff_dim - need;
        for (int i = 0; i < fd; i++) {
            for (int j = 0; j < ld; j++) {
                x[i*(ld+_xappendone)+j] = _xop(_tx[i*ld+j], _tx[i*ld+j]);
            }
            if (_xappendone)
                x[i*(ld+1)+ld] = 1;
        }
        memset(y, 0, sizeof(T)*DData<T>::_y_dim*DData<T>::_buff_dim);
        if (_yonehot) {
            for (int i = 0; i < fd; i++) y[i*DData<T>::_y_dim + _ty[i]] = 1.0;
        }else {
            ld = DData<T>::_y_dim;
            for (int i = 0; i < fd; i++) {
                for (int j = 0; j < ld; j++) {
                    y[i*ld+j] = _yop(_ty[i*ld+j], _ty[i*ld+j]);
                }
            }   
        }

        return fd;
	}
};

template<class T>
class DMnistData : public DBinaryData<T, unsigned char, unsigned char, OpScale<T>, OpNop<T> > {
protected:
    int _split;
public:
    DMnistData(std::string path, int split, int buff_dim, int testing, cublasHandle_t handle) : DBinaryData<T, unsigned char, unsigned char, OpScale<T>, OpNop<T> >(OpScale<T>(1.0/256.0), OpNop<T>(), 28*28, 10, true, true, buff_dim, true, testing, handle) {
        _split = split;
        int soffset, eoffset;
        std::string xpath;
        std::string ypath;
		if (path[path.length()-1] != '/') path.append("/");
        if (split&(DData<T>::Train|DData<T>::Validate)) {
            xpath = path+"train-images-idx3-ubyte";
			ypath = path+"train-labels-idx1-ubyte";
			soffset = 0;
			eoffset = 60000;
			if (!(split&DData<T>::Train)) soffset = 50000;
			if (!(split&DData<T>::Validate)) eoffset = 50000;
        }else {
			xpath = path+"t10k-images-idx3-ubyte";
			ypath = path+"t10k-labels-idx1-ubyte";
			soffset = 0;
			eoffset = 10000;
		}
        DBinaryData<T, unsigned char, unsigned char, OpScale<T>, OpNop<T> >::open(xpath.c_str(), ypath.c_str(), 16, 8, soffset, eoffset);
    }

    ~DMnistData() {
        DData<T>::stop();
    }
};

template<class T>
class DDummyData : public DData<T> {
    int _n;
public:
    DDummyData(int n, cublasHandle_t handle) : DData<T>(1, n+1, n+1, 1, false, false, handle) {
        _n = n + 1;
    }
    virtual int fetch(T *&x, T *&y) { return false; }  
    virtual void start() {}
    virtual int instancesPerEpoch() { return 256; }
    virtual bool getData(DMatrix<T> *&x, DMatrix<T>*&y, int batch_size) {
        x = y = new DMatrix<T>(batch_size, _n, DData<T>::_handle);
        x->init(DMatrix<T>::Zero);
        T *data = x->host_data();
        for (int i = 0; i < batch_size; i++) {
            data[i+(i%(_n-1))*batch_size] = 1.0;
            data[i+(_n-1)*batch_size] = 1.0;
        }
        x->host2dev();
        return true;
    }
};

#ifdef USE_MPI
template<class T>
class DParallelMnistData : public DMnistData<T> {
    int _total_soffset;
    int _total_eoffset;
public:
    DParallelMnistData(string path, int N, int rank, int batch_size, cublasHandle_t handle) : DMnistData<T>(path, DData<T>::Train, batch_size, false, handle) {
        int block_size = DMnistData<T>::instancesPerEpoch()/N;
        _total_soffset = DMnistData<T>::_soffset;
        _total_eoffset = DMnistData<T>::_eoffset;
        DMnistData<T>::_soffset = rank*block_size;
        DMnistData<T>::_eoffset = min(DMnistData<T>::_soffset + block_size, DMnistData<T>::_eoffset);
        if (rank == N - 1) DMnistData<T>::_eoffset = _total_eoffset;
        printf("\nNODE%d now work on [%d,%d)\n", mpi_world_rank, DMnistData<T>::_soffset, DMnistData<T>::_eoffset);
    }

    int totalInstancesPerEpoch() {
        return _total_eoffset - _total_soffset;
    }

    void balance(int s, int t, int sec) {
        if (mpi_world_rank >= s && mpi_world_rank < t) {
            int *buf = new int[t-s];
            MPI_Allgather(&sec, 1, MPI_INT, buf, 1, MPI_INT, MPI_COMM_WORLD);
            float sum = 0;
            for (int i = s; i < t; i++) 
                sum += 1.0/buf[i-s];
            float fs = 0, fe = 0;
            for (int i = s; i <= mpi_world_rank; i++) {
                fs = fe;
                fe += (1.0/buf[i-s])/sum;
            }
            DMnistData<T>::_soffset = fs * (_total_eoffset - _total_soffset) + _total_soffset;
            DMnistData<T>::_eoffset = fe * (_total_eoffset - _total_soffset) + _total_soffset;
            if (mpi_world_rank == t - 1) DMnistData<T>::_eoffset = _total_eoffset;
            printf("\n%d %d %d %f %f\n", s, t, mpi_world_rank, sum, (float)(1.0/buf[mpi_world_rank-s]));
            printf("\nNODE%d now work on [%d,%d)\n", mpi_world_rank, DMnistData<T>::_soffset, DMnistData<T>::_eoffset);
        }
    }
};
#endif

#endif //DDATA_CUH
