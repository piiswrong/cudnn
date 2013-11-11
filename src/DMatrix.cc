#include <DMatrix.cuh>
#include <random>

template<class T>
void DMatrix<T>::init(int p, T a, T b) { 
    if (p&DMatrix<T>::Zero) memset(_host_data, 0, _size);
    if (p&DMatrix<T>::One) for (int i = 0; i < _nelem; i++) _host_data[i] = (T)1;
    if (p&DMatrix<T>::Uniform) {
        std::default_random_engine gen;
        std::uniform_real_distribution<T> dist(a, b);
        for (int i = 0; i < _nelem; i++) _host_data[i] = dist(gen);
    }
    if (p&DMatrix<T>::Normal) {
        std::default_random_engine gen;
        std::normal_distribution<T> dist(a, b);
        for (int i = 0; i < _nelem; i++) _host_data[i] = dist(gen);
    }
    if (p&DMatrix<T>::ColSparse) {
        for (int col = 0; col < _fd; col++) {
            int n1 = SPARSE_DEGREE, n2 = _ld - SPARSE_DEGREE;
            for (int row = 0; row < _ld; row++) {
                int r = rand()%(n1+n2);
                if (r<n1) {
                    n1--;
                }else {
                    n2--;
                    _host_data[col*_ld + row] = 0.0;
                }
            }
        }
    }
    if (p&DMatrix<T>::RowSparse) {
        for (int row = 0; row < _ld; row++) {
            int n1 = SPARSE_DEGREE, n2 = _fd - SPARSE_DEGREE;
            for (int col = 0; col < _fd; col++) {
                int r = rand()%(n1+n2);
                if (r<n1) {
                    n1--;
                }else {
                    n2--;
                    _host_data[col*_ld + row] = 0.9;
                }
            }
        }
    }
    if (p&DMatrix<T>::Weight) {
        for (int i = _nelem - _ld; i < _nelem; i++) _host_data[i] = 0.0;
        _host_data[_nelem-1] = 1.0;
    }
    if (_on_device) host2dev();
}

