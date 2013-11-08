#include <DNN.cuh>
#include <DData.cuh>

int main() {
    DNN<float> dnn;
    DMnistData<float> data('../data', DData::Split::Train, dnn.handle());
    dnn.


    return 0;
}
