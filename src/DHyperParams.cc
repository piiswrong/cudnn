#include "DHyperParams.h"
#include <boost/program_options.hpp>

template<class T>
void reg_param_helper(po::options_description &desc, std::string s, T *x) {
    desc.add_options()                          
        (s.c_str(), po::value<T>(x)->default_value(*x), "set hyper-parameter value") 
    ;                                           
}

#define REG_PARAM(x)                            \
do {                                            \
    std::string s = prefix+"_"+#x;              \
    std::replace(s.begin(), s.end(), '_', '-'); \
    reg_param_helper(desc, s, &x);               \
}while(false)

DHyperParams::DHyperParams() {
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
}

void DHyperParams::registerParams(po::options_description &desc, std::string prefix) {
        REG_PARAM(learning_rate);
        REG_PARAM(learning_rate_decay);
        REG_PARAM(momentum);
        REG_PARAM(max_momentum);
        REG_PARAM(step_momentum);
        REG_PARAM(weight_decay);
        REG_PARAM(decay_rate);
        REG_PARAM(idrop_out);
        REG_PARAM(idrop_rate);
        REG_PARAM(hdrop_out);
        REG_PARAM(hdrop_rate);
        REG_PARAM(batch_size);
        REG_PARAM(check_interval);
}
