#ifndef DLAYER_CUH
#define DLAYER_CUH

#include <time.h>
#include <common.cuh>
#include <DHyperParams.h>
#include <DOperators.cuh>
#include <DNeuron.cuh>
#include <kernels.cuh>



template<class T>
class DLayer {
public:
    DLayer() {}
    ~DLayer() {}
    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {}
    virtual DMatrix<T> *delta() = 0;
    virtual DMatrix<T> *act() = 0;
    virtual DNeuron<T> *neuron() = 0;
    virtual DDim4 output_dim() = 0;
    virtual void clearMomentum() {}

    virtual void fprop(DMatrix<T>* dev_data, bool drop_out, float drop_rate) = 0; 
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* pre_act, float rate, float mom, bool drop_out, bool decay, float decay_rate, bool rectify_weight, bool rectify_bias) = 0; 

    virtual void scaleWeight(float scale) {}
    virtual void regParams(std::vector<DMatrix<T>*> &X, std::vector<DMatrix<T>*> &dX) {}
    void save(FILE *fout) {}
    void load(FILE *fin) {}
};


#endif //DLAYER_CUH
