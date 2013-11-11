#ifndef DRAND_H
#define DRAND_H

class DRand {
    void *gen;
    void *dist_uniform;
    void *dist_normal;
public:
    DRand(float a, float b);
    float normal();
    float uniform();
};


#endif //DRAND_H
