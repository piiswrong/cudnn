#ifndef DDATASPEC_H
#define DDATASPEC_H

#include <string>

namespace boost {
namespace program_options {
    class options_description;
};
};
namespace po = boost::program_options;


class DDataSpec {
public:
    int input_dim, output_dim;
    bool xappendone, yonehot;
    std::string train_data, train_label, test_data, test_label;
    float scaling;
    int train_soffset, train_items, test_soffset, test_items;
    int train_data_skip, train_label_skip, test_data_skip, test_label_skip;

    DDataSpec(); 
    void registerParams(po::options_description &desc);
};

#endif //DDATASPEC_H
