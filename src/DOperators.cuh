#ifndef DOPERATORS_CUH
#define DOPERATORS_CUH

#include <common.cuh>

template<class T>
class OpMul{
public:
    HOSTDEVICE T operator() (T x, T y) {
        return x*y;
    }
};

template<class T>
class OpADMMDecay{
    const T _scale;
public:
    OpADMMDecay(const T scale) : _scale(scale) {}
    HOSTDEVICE T operator() (T x, T y, T z) {
        return x+(y+z)*_scale;
    }
};

template<class T>
class OpDPSGDMom{
    const T _scale;
public:
    OpDPSGDMom(const T scale) : _scale(scale) {}
    HOSTDEVICE T operator() (T x, T y) {
        return y+x*_scale;
    }
};

template<class T>
class OpScaleAdd{
    const T _scale;
public:
    OpScaleAdd(const T scale) : _scale(scale) {}
    HOSTDEVICE T operator() (T x, T y) {
        return x+y*_scale;
    }
};

template<class T>
class OpScale{
    const T _scale;
public:
    OpScale(const T scale) : _scale(scale) {}
    HOSTDEVICE T operator() (T x, T y) {
        return y*_scale;
    }
};


template<class T>
class OpWeightedLog{
public:
    HOSTDEVICE T operator() (T x, T y, T z) {
        return log(y)*z;
    }
};

template<class T>
class OpWeighted{
public:
    HOSTDEVICE T operator() (T x, T y, T z) {
        return y*z;
    }
};


template<class T>
class OpLog{
public:
    HOSTDEVICE T operator() (T x, T y) {
        return log(y);
    }
};

template<class T>
class OpExp{
public:
    HOSTDEVICE T operator() (T x, T y) {
        return y*y;
    }
};


template<class T>
class OpSqr{
public:
    HOSTDEVICE T operator() (T x, T y) {
        return exp(y);
    }
    HOSTDEVICE T operator() (T y) {
        return exp(y);
    }
};

template<class T>
class OpAdd{
public:
    HOSTDEVICE T operator() (T x, T y, T z) {
        return y + z;
    }
};

template<class T>
class OpSubEqu{
public:
    HOSTDEVICE T operator() (T x, T y, T z) {
        return x + y - z;
    }
};


template<class T>
class OpSub{
public:
    HOSTDEVICE T operator() (T x, T y, T z) {
        return y - z;
    }
};

template<class T>
class OpNop {
public:
    HOSTDEVICE T operator() (T y) {
        return y;
    }
    HOSTDEVICE T operator() (T x, T y) {
        return y;
    }
};

template<class T>
class OpMinReduce {
public:
    const T Unit;
    HOSTDEVICE OpMinReduce() : Unit(1e37) {}
    HOSTDEVICE T operator() (T x, int i, T y, int j, int &ind) {
        if (x >= y) {
            ind = j;
            return y;
        }else {
            ind = i;
            return x;
        }
    }
};

template<class T>
class OpMaxReduce {
public:
    const T Unit;
    HOSTDEVICE OpMaxReduce() : Unit(-1e37) {}
    HOSTDEVICE T operator() (T x, int i, T y, int j, int &ind) {
        if (x < y) {
            ind = j;
            return y;
        }else {
            ind = i;
            return x;
        }
    }
};

template<class T>
class OpSumReduce {
public:
    const T Unit;
    HOSTDEVICE OpSumReduce() : Unit(0.0) {}
    HOSTDEVICE T operator() (T x, int i, T y, int j, int &ind) {
        return x + y;
    }
};

template<class T>
class OpSubExp{
    const T _shift;
public:
    HOSTDEVICE OpSubExp(T shift) : _shift(shift) {}
    HOSTDEVICE T operator() (T x) {
        return exp(x-_shift);
    }
};



#endif //DOPERATORS_CUH
