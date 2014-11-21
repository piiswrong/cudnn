#ifndef DOPTION_H
#define DOPTION_H

#include <string>
#include <DDataSpec.h>
#include <DHyperParams.h>

namespace boost {
    namespace program_options {
        class variables_map;
    }
};


class DOption {
private:
    boost::program_options::variables_map *_vm;

public:
    std::string master_file, log_path, input_path, output_path, pred_path, data_spec_path;
    int log_verbosity;
    int resuming, devId;
    bool grad_check;
    int num_layers, hidden_dim, input_dim, output_dim;
    std::string net_spec;
    std::string neuron;
    float pt_epochs;
    int bp_epochs;
    DHyperParams bp_hyper_params, pt_hyper_params;
    DDataSpec data_spec;

    DOption();
    int count(std::string s);
    bool parse(int argc, char **argv);
};

#endif //DOPTION_H
