#include <cstdlib>
#include <cstdio>
#include <tr1/random>
#include <time.h>
#include <stdint.h>

float abs(float x) {
    const uint32_t mask = (1<<31)-1;
    uint32_t &i = (uint32_t&)x;
    i &= mask;
    return x;
}

template <int n>
inline float nth_rootf(float x)
{
    const uint32_t ebits = 8;
    const uint32_t fbits = 23;
    const uint32_t mask  = (1<<31)-1;
        
    int32_t& i = (int32_t&) x;
    const uint32_t bias = (1 << (ebits-1))-1;
    if (abs(x)>1.0); {
        i = (((i&mask) - (bias << fbits)) / n + (bias << fbits))|(i&(~mask));
    }
        

    return x;
}

inline float newton(float y, int &n) {
    float x = nth_rootf<3>(y);
    float x0;    
    n = 0;
    do {
        x0 = x;
        //x =(2.0/3.0)*x + ( -(2.0/3.0)*x0  + y )/(3*x0*x0+1);
        x = (2.0*x*x*x + y)/(3*x*x+1);
        n++;
    }while (abs((x-x0)/x)>1e-6);
    return x;
}

int main() {
/*    int nn;
    for (float i = -10.0; i < 10.0; i += 0.1) {
        printf("%f %f %f ", i, nth_rootf<3>(i), newton(i,nn));
    }
    return 0;
*/
    srand(time(0));
    int n;
    int total=0;
    float error = 0;
    int N =1000000;
    int t = clock();
    for (int i = 0; i < N; i++) {
        float y = rand();
        y = (y/RAND_MAX)*10-5;
        float x = newton(y, n);
        error += abs((x*x*x+x-y));
        total += n;
    }
    printf("%f\n%d\n%f\n", ((float)total)/N, clock()-t, error/N);

}
