#ifndef DHYPERPARAMS_CUH
#define DHYPERPARAMS_CUH

#include <string>

namespace boost {
namespace program_options {
    class options_description;
};
};
namespace po = boost::program_options;

class DHyperParams {
public:
    float learning_rate;
    float current_learning_rate;
    float learning_rate_decay;
    float momentum;
    float current_momentum;
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

    DHyperParams();
    void registerParams(po::options_description &desc, std::string prefix);
};

#endif //DHYPERPARAMS_CUH
