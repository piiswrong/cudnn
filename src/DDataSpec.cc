#include "DDataSpec.h"
#include <boost/program_options.hpp>

DDataSpec::DDataSpec() {
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

void DDataSpec::registerParams(po::options_description &desc) {
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

