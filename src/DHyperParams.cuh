#ifndef DHYPERPARAMS_CUH
#define DHYPERPARAMS_CUH

class DHyperParams {
public:
    float learning_rate;
    float momentum;
    bool weight_decay;
    float decay_rate;
    bool idrop_out;
    float idrop_rate
    bool hdrop_out;
    float hdrop_rate;
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
