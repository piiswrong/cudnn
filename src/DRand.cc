#include <random>
#include <DRand.h>

DRand::DRand(float a, float b) {
    gen = (void*) new std::default_random_engine();
    dist_uniform = (void*) new std::uniform_real_distribution<float>(a, b);
    dist_normal = (void*) new std::normal_distribution<float>(a, b);
}

float DRand::uniform() {
    std::default_random_engine& _gen = *(std::default_random_engine*)gen;
    std::uniform_real_distribution<float>& _dist = *(std::uniform_real_distribution<float>*)dist_uniform;
    return _dist(_gen);
}

float DRand::normal() {
    std::default_random_engine& _gen = *(std::default_random_engine*)gen;
    std::normal_distribution<float>& _dist = *(std::normal_distribution<float>*)dist_normal;
    return _dist(_gen);
}
