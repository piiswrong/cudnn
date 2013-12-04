#ifndef DNN_CUH
#define DNN_CUH

#include <common.cuh>
#include <DData.cuh>
#include <DLayer.cuh>
#include <DOperators.cuh>
#include <DHyperParams.cuh>


template<class T>
class DNN {
    bool _on_device;
    cublasHandle_t _handle;
    int _num_layers;
    int *_layer_dims;
    DHyperParams _pt_hyper_params;
    DHyperParams _bp_hyper_params;
    DLayer<T>** _layers;
    DMatrix<T>* _delta;
    curandState *_state;
    bool _scaled_weight;

public:
    DNN(int num_layers, int *layer_dims, DNeuron<T> **neurons, 
        DHyperParams& pt_hyper_params, DHyperParams& bp_hyper_params,
        cublasHandle_t handle = 0) {
        _handle = handle;
        if (_handle != 0) {
            _on_device = true;
        }else {
            _on_device = false;
        }
        _num_layers = num_layers;
        _layer_dims = layer_dims;
        _pt_hyper_params = pt_hyper_params;
        _bp_hyper_params = bp_hyper_params;
        _scaled_weight = false;

        _layers = new DLayer<T>*[_num_layers];
        for (int i = 0; i < _num_layers; i++) {
            _layers[i] = new DLayer<T>(layer_dims[i], layer_dims[i+1], i, neurons[i],
                                    &_pt_hyper_params, &_bp_hyper_params, _handle);
        }
        _delta = new DMatrix<T>(_bp_hyper_params.batch_size, layer_dims[_num_layers]+1, _handle);
        _delta->init(DMatrix<T>::Zero);
#ifndef DISABLE_GPU
        if (_on_device) {
            int nstate = (_layer_dims[0]+1)*_bp_hyper_params.batch_size;
            CUDA_CALL(cudaMalloc((void**)&_state, nstate*sizeof(curandState)));
            dim3 grid((nstate-1)/BLOCK_SIZE+1);
            dim3 block(BLOCK_SIZE);
            kSetupCurand<<<grid, block>>>(_state, nstate, time(0));
        }
#endif
    }

    cublasHandle_t handle() { return _handle; }

    void save(FILE *fout) {

    }

    T trainOnBatch(DMatrix<T>* x, DMatrix<T>* y) {
        fprop(x, _num_layers, _layers, _bp_hyper_params.idrop_out, _bp_hyper_params.hdrop_out,
                _bp_hyper_params.idrop_rate, _bp_hyper_params.hdrop_rate);
        return bprop(x, y, _num_layers, _layers);
    }

    void fprop(DMatrix<T>* x, int num_layers, DLayer<T>** layers, bool idrop_out = false,
                        bool hdrop_out = false, float idrop_rate = 0.0, float hdrop_rate = 0.0) {
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) {
#endif
            if (idrop_out) hDropout<T>(x, NULL, _state, idrop_rate, x->getT(), x->nrows(), x->ncols() - 1, x->ld());
            layers[0]->fprop(x, num_layers > 1 && hdrop_out, hdrop_rate);
            for (int i = 1; i < num_layers; i++) 
                layers[i]->fprop(layers[i-1]->act(), i < num_layers - 1 && hdrop_out, hdrop_rate);
#ifdef DOWN_POUR_SGD
        }
#endif
    }

    T bprop(DMatrix<T>* x, DMatrix<T>* y, int num_layers, DLayer<T>** layers) {
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) 
#endif
            layers[num_layers-1]->neuron()->initDelta(_delta, layers[num_layers-1]->act(), y);
        DMatrix<T>* d = _delta;
        for (int i = num_layers-1; i > 0; i--) {
            layers[i]->bprop(d, layers[i-1]->act(), _bp_hyper_params.learning_rate, _bp_hyper_params.momentum,
                            _bp_hyper_params.hdrop_out, _bp_hyper_params.weight_decay, _bp_hyper_params.decay_rate);
            d = layers[i]->delta();
        }
        layers[0]->bprop(d, x, _bp_hyper_params.learning_rate, _bp_hyper_params.momentum,
                            _bp_hyper_params.hdrop_out, _bp_hyper_params.weight_decay, _bp_hyper_params.decay_rate);
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) {
            layers[num_layers-1]->neuron()->computeLoss(_delta, layers[num_layers-1]->act(), y);
            T loss = layers[num_layers-1]->neuron()->getLoss();
            MPI_Send(&loss, 1, _delta->mpiDatatype(), 0, SGD_LOSS_TAG, MPI_COMM_WORLD);
            return loss;
        }else {
            T loss;
            MPI_Status status;
            MPI_Recv(&loss, 1, _delta->mpiDatatype(), MPI_ANY_SOURCE, SGD_LOSS_TAG, MPI_COMM_WORLD, &status);
            return loss;
        }
#else
        layers[num_layers-1]->neuron()->computeLoss(_delta, layers[num_layers-1]->act(), y);
        return layers[num_layers-1]->neuron()->getLoss();
#endif
    }

    void scaleWeight(bool scaled_weight) {
        if (scaled_weight == _scaled_weight) return;
        if (!_scaled_weight) {
            if (_bp_hyper_params.idrop_out)
                _layers[0]->scaleWeight(1-_bp_hyper_params.idrop_rate);
            if (_bp_hyper_params.hdrop_out)
                for (int i = 1; i < _num_layers; i++)
                    _layers[i]->scaleWeight(1-_bp_hyper_params.hdrop_rate);
            _scaled_weight = true;
        }else {
            if (_bp_hyper_params.idrop_out)
                _layers[0]->scaleWeight(1.0/(1-_bp_hyper_params.idrop_rate));
            if (_bp_hyper_params.hdrop_out)
                for (int i = 1; i < _num_layers; i++)
                    _layers[i]->scaleWeight(1.0/(1-_bp_hyper_params.hdrop_rate));
            _scaled_weight = false;
        }
    }

#ifdef ADMM
    void admmReduce() {
        for (int i = 0; i < _num_layers; i++) {
            _layers[i]->ADMM_reduce();
        }
    }
    
    void admmFineTune(DData<T> *data, int total_epochs) {
        for (int epoch = 0; epoch < total_epochs; epoch += _bp_hyper_params.reduce_epochs) {
            T error = fineTune(data, _bp_hyper_params.reduce_epochs);
            int n = data->instancesPerEpoch();
            admmReduce();
            T total_error;
            int total_n;
            error *= n;
            MPI_Reduce(&n, &total_n, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&error, &total_error, 1, _delta->mpiDatatype(), MPI_SUM, 0, MPI_COMM_WORLD);
            if (mpi_world_rank == 0) {
                printf("\n*************************************\n"
                         "Iteration: %d\tError: %f\n"
                         "*************************************\n", epoch/_bp_hyper_params.reduce_epochs, (float)(total_error/total_n));
            }
        }
    }
#endif

    T test(DData<T>* data) {
        scaleWeight(true);
        data->start();
        int iperEpoch = data->instancesPerEpoch();
        DMatrix<T> *x, *y;
        T loss = 0.0;
        CUDA_CALL(cudaThreadSynchronize());
        while (true) {
            bool more = data->getData(x, y, _bp_hyper_params.batch_size);
            fprop(x, _num_layers, _layers);
            _layers[_num_layers-1]->neuron()->initDelta(_delta, _layers[_num_layers-1]->act(), y);
            _layers[_num_layers-1]->neuron()->computeLoss(_delta, _layers[_num_layers-1]->act(), y);
            loss += _layers[_num_layers-1]->neuron()->getLoss();
            CUDA_CALL(cudaThreadSynchronize());
            if (!more) break;
        }
        return loss/iperEpoch;
    }

    T fineTune(DData<T>* data, int total_epochs) {
        scaleWeight(false);
        data->start();
        int iperEpoch = data->instancesPerEpoch();
        DMatrix<T> *x, *y;
        int nEpoch = 1;
        int nInstance = 0;
        T error = 0.0;
        bool checked = false;
        T lastError;
        int lastCheck = 0;
        
        CUDA_CALL(cudaThreadSynchronize());
        while ( nEpoch <= total_epochs ) {
            data->getData(x, y, _bp_hyper_params.batch_size);
            error += trainOnBatch(x, y);
            CUDA_CALL(cudaThreadSynchronize());
            nInstance += _bp_hyper_params.batch_size;
            while (nEpoch*iperEpoch <= nInstance) {
                nEpoch++;
                _bp_hyper_params.learning_rate *= _bp_hyper_params.learning_rate_decay;
                _bp_hyper_params.momentum += _bp_hyper_params.step_momentum;
                if (_bp_hyper_params.momentum > _bp_hyper_params.max_momentum)
                    _bp_hyper_params.momentum = _bp_hyper_params.max_momentum;
            }
            lastCheck += _bp_hyper_params.batch_size;
            if (lastCheck >= _bp_hyper_params.check_interval) {
                DMatrix<T> *m;
/*#ifndef NDEBUG
                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = x;//_layers[i]->delta();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                m = _delta;
                m->dev2host();
                for (int r = 0; r < 10; r++) {
                    for (int c = 0; c < m->ncols(); c++) {
                        printf("%+1.3f ", (float)m->getElem(r,c));
                    }
                    printf("\n");
                }
                printf("\n");

                m = y;//_layers[i]->delta();
                m->dev2host();
                for (int r = 0; r < 10; r++) {
                    for (int c = 0; c < m->ncols(); c++) {
                        printf("%+1.3f ", (float)m->getElem(r,c));
                    }
                    printf("\n");
                }
                printf("\n");
                
                m = _layers[_num_layers-1]->drv();
                m->dev2host();
                for (int r = 0; r < 10; r++) {
                    for (int c = 0; c < m->ncols(); c++) {
                        printf("%+1.3f ", (float)m->getElem(r,c));
                    }
                    printf("\n");
                }
                printf("\n");
                
                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = _layers[i]->drv();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }*/
/*
                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = _layers[i]->act();
                    m->dev2host();
                    for (int r = 0; r < 10; r++) {
                        for (int c = 0; c < m->ncols(); c++) {
                            printf("%+1.3f ", (float)m->getElem(r,c));
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                for (int i = 0; i < _num_layers; i++) {
                    DMatrix<T> *m = _layers[i]->weight();
                    m->dev2host();
                    for (int r = 0; r < m->ld(); r++) {
                        for (int c = 0; c < m->fd(); c++) {
                            printf("%+1.3f ", (float)m->host_data()[r+c*m->ld()]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
#endif*/
#ifdef USE_MPI
                printf("\nNode%d\tEpoch: %d\tInstance: %d\tError: %f\n", mpi_world_rank, nEpoch, nInstance%iperEpoch, (float)(error/lastCheck));
#else
                printf("\nEpoch: %d\tInstance: %d\tError: %f\n", nEpoch, nInstance%iperEpoch, (float)(error/lastCheck));
#endif
                checked = true;
                lastError = error/lastCheck;
                lastCheck = 0;
                error = 0.0;
            }
            delete x, y;
        }
        if (checked) {
            return lastError;
        }else {
            return error / lastCheck;
        }
    }

};


#endif //DNN_CUH
