#ifndef DHYPERPARAMS_CUH
#define DHYPERPARAMS_CUH

class DHyperParams {
public:
    float learning_rate;
    float learning_rate_decay;
    float momentum;
    float max_momentum;
    float step_momentum;
    bool weight_decay;
    float decay_rate;
    bool idrop_out;
    float idrop_rate;
    bool hdrop_out;
    float hdrop_rate;
    int batch_size;
    int check_interval;

    DHyperParams() {
        learning_rate = 0.1;
        learning_rate_decay = 0.995;
        momentum = 0.5;
        max_momentum = 0.99;
        step_momentum = 0.01;
        weight_decay = false;
        decay_rate = 1e-6;
        idrop_out = false;
        idrop_rate = 0.5;
        hdrop_out = false;
        hdrop_rate = 0.5;
        batch_size = 128;
        check_interval = 10000;
    }
};

#endif //DHYPERPARAMS_CUH
