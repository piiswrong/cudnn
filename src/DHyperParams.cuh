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
    bool sparseInit;
    int batch_size;
    int check_interval;
#ifdef ADMM
    int reduce_epochs;
#endif

    DHyperParams() {
        learning_rate = 1.0;
        learning_rate_decay = 0.998;
        momentum = 0.5;
        max_momentum = 0.90;
        step_momentum = 0.40;
        weight_decay = false;
        decay_rate = 1e-6;
        idrop_out = true;
        idrop_rate = 0.2;
        hdrop_out = true;
        hdrop_rate = 0.5;
        batch_size = 128;
        sparseInit = false;
        check_interval = 10000;
#ifdef ADMM
        reduce_epochs = 5;
#endif
    }
};

#endif //DHYPERPARAMS_CUH
