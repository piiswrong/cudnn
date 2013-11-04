#ifndef DHYPERPARAMS_CUH
#define DHYPERPARAMS_CUH

template<class T>
class DHyperParams {
public:
    T learning_rate;
    T momentum;
    bool weight_decay;
    T decay_rate;
    bool idrop_out;
    T idrop_rate
    bool hdrop_out;
    T hdrop_rate;
    int batch_size;

    DHyperParams() {
        learning_rate = 0.01;
        momentum = 0.9;
        weight_decay = false;
        decay_rate = 1e-6;
        idrop_out = false;
        idrop_rate = 0.5;
        hdrop_out = false;
        hdrop_rate = 0.5;
        batch_size = 128;
    }
}

#endif //DHYPERPARAMS_CUH
