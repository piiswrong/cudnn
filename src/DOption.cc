#include "DOption.h"
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
namespace po = boost::program_options;

DOption::DOption() {
    _vm = new po::variables_map;
}

int DOption::count(std::string s) {
    return _vm->count(s);
}

bool DOption::parse(int argc, char **argv) {
    po::options_description generic_opts("Generic options");
    po::options_description hyper_opts("Hyper-parameters");
    po::options_description data_opts("Data set options");
    po::options_description cmdline_opts;
    po::options_description master_file_opts;
    po::positional_options_description p;
    try{
        generic_opts.add_options()
            ("help", "print this help massage")
            ("master", po::value<std::string>(&master_file), "path to master file")
            ("data", po::value<std::string>(&data_spec_path)->default_value("../../paris/data_spec"), "path to data specification file")
            ("resume,r", po::value<int>(&resuming)->default_value(-1), "resume from n-th epoch")
            ("device,d", po::value<int>(&devId)->default_value(-1), "id of GPU to use")
            ("check-grad,c", po::value<bool>(&grad_check)->default_value(false)->implicit_value(true), "performe numerical gradient checking")
            ("log-path,l", po::value<std::string>(&log_path)->default_value(""), "path to log file")
            ("input-path,i", po::value<std::string>(&input_path), "path to input network parameter file")
            ("output-path,o", po::value<std::string>(&output_path), "path to output network parameter file")
            ("test,t", po::value<std::string>(&pred_path)->implicit_value(""), "performe test and output predictions to file if specified")
            ("verbose,v", po::value<int>(&log_verbosity), "log output verbosity")
        ;

        hyper_opts.add_options()
            ("num-layers", po::value<int>(&num_layers)->default_value(num_layers), "set hyper-paramter value")
            ("hidden-dim", po::value<int>(&hidden_dim)->default_value(hidden_dim), "set hyper-paramter value")
            ("neuron", po::value<std::string>(&neuron)->default_value(neuron), "set hyper-paramter value")
            ("pt-epochs", po::value<float>(&pt_epochs)->default_value(pt_epochs), "set hyper-paramter value")
            ("bp-epochs", po::value<int>(&bp_epochs)->default_value(bp_epochs), "set hyper-paramter value")
        ;
        pt_hyper_params.registerParams(hyper_opts, "pt");
        bp_hyper_params.registerParams(hyper_opts, "bp");

        data_spec.registerParams(data_opts);
        
        cmdline_opts.add(generic_opts).add(hyper_opts);
        master_file_opts.add(hyper_opts);
        p.add("master", -1);

        store(po::command_line_parser(argc, argv).
            options(cmdline_opts).positional(p).run(), *_vm);
        notify(*_vm);

        if (_vm->count("help")) {
            std::cout << cmdline_opts << "\n";
            exit(0);
        }

        if (_vm->count("master")) {
            if (!_vm->count("log-path")) log_path = master_file+".log";
            if (!_vm->count("input-path")) input_path = master_file+".param";
            if (!_vm->count("output-path")) output_path = master_file+".param";
            std::ifstream ifs(master_file.c_str());
            if (!ifs) {
                printf("Cannot open master file %s\n", master_file.c_str());
                exit(-1);
            }
            printf("Using master file %s\n", master_file.c_str());
            store(parse_config_file(ifs, master_file_opts), *_vm);
            notify(*_vm);
        }
        if (_vm->count("data")) {
            std::ifstream ifs(data_spec_path.c_str());
            if (!ifs) {
                printf("Cannot open data specification file %s\n", master_file.c_str());
                exit(-1);
            }
            printf("Using data set %s\n", data_spec_path.c_str());
            store(parse_config_file(ifs, data_opts), *_vm);
            notify(*_vm);
            input_dim = data_spec.input_dim;
            //output_dim = data_spec.output_dim;
            fs::path base(data_spec_path);
            base.remove_filename();
            data_spec.train_data = canonical(data_spec.train_data, base).c_str();
            data_spec.train_label = canonical(data_spec.train_label, base).c_str();
            data_spec.test_data = canonical(data_spec.test_data, base).c_str();
            data_spec.test_label = canonical(data_spec.test_label, base).c_str();
        }

    }catch(std::exception &e) {
        std::cout << "Error: " << e.what() << "\n";
        std::cout << cmdline_opts << "\n";
        std::cout << data_opts << "\n";
        return false;
    }
    return true;
}
