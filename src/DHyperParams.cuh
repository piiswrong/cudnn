#ifndef DHYPERPARAMS_CUH
#define DHYPERPARAMS_CUH

#include <common.cuh>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define REG_PARAM(x)                            \
do {                                            \
    std::string s = prefix+"_"+#x;              \
    std::replace(s.begin(), s.end(), '_', '-'); \
    regParam(desc, s, &x);                      \
}while(false)

class DHyperParams {
private:
    template<class T>
    void regParam(po::options_description &desc, std::string name, T *x) {
        desc.add_options()
            (name.c_str(), po::value<T>(x)->default_value(*x), "set hyper-parameter value")
        ;
    }
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

    void registerParams(po::options_description &desc, std::string prefix) {
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
};

class DDataSpec {
public:
    int input_dim, output_dim;
    bool xappendone, yonehot;
    std::string train_data, train_label, test_data, test_label;
    float scaling;
    int train_soffset, train_items, test_soffset, test_items;
    int train_data_skip, train_label_skip, test_data_skip, test_label_skip;

    DDataSpec() {
        xappendone = true;
        yonehot = true;
        scaling = 1.0;
        train_soffset = 0;
        test_soffset = 0;
        train_data_skip = 0;
        train_label_skip = 0;
        test_data_skip = 0;
        test_label_skip = 0;
    }

    void registerParams(po::options_description &desc) {
        desc.add_options() 
            ("input-dim", po::value<int>(&input_dim)->required(), "Set data set parameter")
            ("output-dim", po::value<int>(&output_dim)->required(), "Set data set parameter")
            ("xappendone", po::value<bool>(&xappendone)->default_value(xappendone), "Set data set parameter")
            ("yonehot", po::value<bool>(&yonehot)->default_value(yonehot), "Set data set parameter")
            ("train-data", po::value<std::string>(&train_data)->required(), "Set data set parameter")
            ("train-label", po::value<std::string>(&train_label)->required(), "Set data set parameter")
            ("test-data", po::value<std::string>(&test_data)->required(), "Set data set parameter")
            ("test-label", po::value<std::string>(&test_label)->required(), "Set data set parameter")
            ("scaling", po::value<float>(&scaling)->default_value(scaling), "set data set parameter")
            ("train-soffset", po::value<int>(&train_soffset)->default_value(train_soffset), "Set data set parameter")
            ("train-items", po::value<int>(&train_items)->required(), "Set data set parameter")
            ("test-soffset", po::value<int>(&test_soffset)->default_value(test_soffset), "Set data set parameter")
            ("test-items", po::value<int>(&test_items)->required(), "Set data set parameter")
            ("train-data-skip", po::value<int>(&train_data_skip)->default_value(train_data_skip), "Set data set parameter")
            ("train-label-skip", po::value<int>(&train_label_skip)->default_value(train_label_skip), "Set data set parameter")
            ("test-data-skip", po::value<int>(&test_data_skip)->default_value(test_data_skip), "Set data set parameter")
            ("test-label-skip", po::value<int>(&test_label_skip)->default_value(test_label_skip), "Set data set parameter")
        ;
    }

};

#endif //DHYPERPARAMS_CUH
