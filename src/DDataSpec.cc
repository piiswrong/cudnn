#include "DDataSpec.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
namespace po = boost::program_options;

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
        ("input-dim", po::value<std::vector<int> >(&_input_dims)->multitoken()->required(), "Set data set parameter")
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

void DDataSpec::parse(std::string data_spec_path) {
    po::variables_map vm;
    po::options_description data_opts("Data set options");
    registerParams(data_opts);

    std::ifstream ifs(data_spec_path.c_str());
    if (!ifs) {
        printf("Cannot open data specification file %s\n", data_spec_path.c_str());
        exit(-1);
    }
    printf("Using data set %s\n", data_spec_path.c_str());
    store(parse_config_file(ifs, data_opts), vm);
    notify(vm);
    if (_input_dims.size() == 1) {
        input_4d = DDim4(0, 1, 1, input_dim);
    }else if (_input_dims.size() == 3) {
        input_4d = DDim4(0, _input_dims[0], _input_dims[1], _input_dims[2]);
    }else {
        std::cout << "Invalid input dimensions ";
        for (int i = 0; i < _input_dims.size(); i++) std::cout << _input_dims[i] << " ";
        std::cout << std::endl;
        exit(-1);
    }
    input_dim = input_4d.c*input_4d.h*input_4d.w;

    fs::path base(data_spec_path);
    base.remove_filename();
    train_data = canonical(train_data, base).c_str();
    train_label = canonical(train_label, base).c_str();
    test_data = canonical(test_data, base).c_str();
    test_label = canonical(test_label, base).c_str();
}

