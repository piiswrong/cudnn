#ifndef DHYPERPARAMS_CUH
#define DHYPERPARAMS_CUH

#include <common.cuh>

#define READ_PARAM(x) do { float tmp; fscanf(fin, #x "=%f\n", &tmp); x = tmp; printf(#x "=%f\n", (float)x); LOG(VERBOSE_NORMAL, fprintf(flog, #x "=%f\n", (float)x)); }while(false)

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
#ifdef ADMM
    int reduce_epochs;
#endif

    DHyperParams() {
        learning_rate = 0.1;
        learning_rate_decay = 0.998;
        momentum = 0.5;
        max_momentum = 0.90;
        step_momentum = 0.04;
        weight_decay = false;
        decay_rate = 1e-6;
        idrop_out = false;
        idrop_rate = 0.2;
        hdrop_out = false;
        hdrop_rate = 0.5;
        batch_size = 128;
        check_interval = 10000;
#ifdef ADMM
        reduce_epochs = 5;
#endif
    }

    void load(FILE *fin) {
        READ_PARAM(learning_rate);
        READ_PARAM(learning_rate_decay);
        READ_PARAM(momentum);
        READ_PARAM(max_momentum);
        READ_PARAM(step_momentum);
        READ_PARAM(weight_decay);
        READ_PARAM(decay_rate);
        READ_PARAM(idrop_out);
        READ_PARAM(idrop_rate);
        READ_PARAM(hdrop_out);
        READ_PARAM(hdrop_rate);
        READ_PARAM(batch_size);
        READ_PARAM(check_interval);
    }
};

#endif //DHYPERPARAMS_CUH
