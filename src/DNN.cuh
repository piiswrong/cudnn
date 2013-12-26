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
            CUDA_KERNEL_CHECK();
        }
#endif
    }

    cublasHandle_t handle() { return _handle; }
    DLayer<T> **layers() { return _layers; }

    void save(FILE *fout) {
        scaleWeight(true);
        fprintf(fout, "%d\n", _num_layers);
        for (int i = 0; i < _num_layers; i++) {
            fprintf(fout, "%d\ng%d ", i, i);
            DMatrix<T> *x = _layers[i]->weight();
            x->dev2host();
            fprintf(fout, "%d %d\n", x->ncols() - 1, x->nrows());
            for (int j = 0; j < x->ncols() - 1; j++) {
                for (int k = 0; k < x->nrows(); k++) 
                    fprintf(fout, "%lf ", (double)x->getElem(k, j));
                fprintf(fout, "\n");
            }
        }
    }

    T trainOnBatch(DMatrix<T>* x, DMatrix<T>* y) {
        fprop(x, _num_layers, _layers, &_bp_hyper_params, _state);
        return bprop(x, y, _num_layers, _layers, &_bp_hyper_params);
    }

    void fprop(DMatrix<T>* x, int num_layers, DLayer<T>** layers, DHyperParams *params, curandState *state) {
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) {
#endif
            if (params->idrop_out) hDropout<T>(x, NULL, state, params->idrop_rate, x->getT(), x->nrows(), x->ncols() - 1, x->ld());
            layers[0]->fprop(x, (num_layers > 1) && params->hdrop_out, params->hdrop_rate);
            for (int i = 1; i < num_layers; i++) {
                layers[i]->fprop(layers[i-1]->act(), (i < num_layers - 1) && params->hdrop_out, params->hdrop_rate);
            }
#ifdef DOWN_POUR_SGD
        }
#endif
    }

    T bprop(DMatrix<T>* x, DMatrix<T>* y, int num_layers, DLayer<T>** layers, DHyperParams *params) {
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) 
#endif
            layers[num_layers-1]->neuron()->initDelta(_delta, layers[num_layers-1]->act(), y);
        DMatrix<T>* d = _delta;
        for (int i = num_layers-1; i > 0; i--) {
            layers[i]->bprop(d, layers[i-1]->act(), params->learning_rate, params->momentum,
                            (i < num_layers-1) && params->hdrop_out, params->weight_decay, params->decay_rate);
            d = layers[i]->delta();
        }
        layers[0]->bprop(d, x, params->learning_rate, params->momentum,
                            (num_layers>1) && params->hdrop_out, params->weight_decay, params->decay_rate);
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server) {
            layers[num_layers-1]->neuron()->computeLoss(_delta, layers[num_layers-1]->act(), y);
            T loss = layers[num_layers-1]->neuron()->getLoss();
            MPI_Send(&loss, 1, _delta->mpiDatatype(), 0, SGD_LOSS_TAG, MPI_COMM_WORLD);
            return loss;
        }else if (mpi_world_rank == 0) {
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
        int starting = time(0);
        bool balanced = false;
        int sec = 0;
        for (int epoch = 0; epoch < total_epochs; epoch += _bp_hyper_params.reduce_epochs) {
            int last_t = time(0);
            T error = fineTune(data, _bp_hyper_params.reduce_epochs);
            sec += time(0) - last_t;
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
                LOG(fprintf(flog, "%f %d\n", (float)(total_error/total_n), time(0)-starting));
            }
            if (mpi_world_size > 1 && epoch >= 9 && !balanced) {
                balanced = true;
                data->stop();
                data->balance(0, mpi_world_size, sec);
                data->start();
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
        DHyperParams dummy_param(_pt_hyper_params);
        _pt_hyper_params.idrop_out = _pt_hyper_params.hdrop_out = false;
        _pt_hyper_params.idrop_rate = _pt_hyper_params.hdrop_rate = false;
        while (true) {
            bool more = data->getData(x, y, _bp_hyper_params.batch_size);
            fprop(x, _num_layers, _layers, &dummy_param, _state);
            _layers[_num_layers-1]->neuron()->initDelta(_delta, _layers[_num_layers-1]->act(), y);
            _layers[_num_layers-1]->neuron()->computeLoss(_delta, _layers[_num_layers-1]->act(), y);
            loss += _layers[_num_layers-1]->neuron()->getLoss();
            CUDA_CALL(cudaThreadSynchronize());
            if (!more) break;
        }
        return loss/iperEpoch;
    }

    void pretrain(DData<T>* data, double epochs_per_layer) {
        DMatrix<T> *x, *y;
        DHyperParams dummy_param(_pt_hyper_params);
        dummy_param.idrop_out = dummy_param.hdrop_out = false;
        dummy_param.idrop_rate = dummy_param.hdrop_rate = false;
        data->start();
        int iperEpoch = data->instancesPerEpoch();
        int instances_per_layer = iperEpoch * epochs_per_layer;
        for( int layer = 1 ; layer < _num_layers; layer++) {
            printf("Pretraining layer %d of %d\n", layer, _num_layers);
            int nEpoch = 1;
            int nInstance = 0;
            int lastCheck = 0;
            T error = 0.0;
            DLayer<T> **layers = new DLayer<T>*[layer+2];
            DMatrix<T> *tmp = new DMatrix<T>(_bp_hyper_params.batch_size, _layer_dims[layer]+1, _handle);
            for (int i = 0; i <= layer; i++) layers[i] = _layers[i];
            layers[layer+1] = new DLayer<T>(_layer_dims[layer+1], _layer_dims[layer], layer + _num_layers, layer>0?layers[layer-1]->neuron():new DNeuron<T>(_handle), 
                                            &_pt_hyper_params, &_bp_hyper_params, _handle);
            while (instances_per_layer > nInstance) {
                CUDA_CALL(cudaThreadSynchronize());
                data->getData(x, y, _bp_hyper_params.batch_size);
                DMatrix<T> *input = x;
                curandState *state = _state;
                if (layer > 0) {
                    fprop(x, layer, layers, &dummy_param, state);
                    input = layers[layer-1]->act();
                    state = layers[layer-1]->state();
                }
                tmp->CopyFrom(input);
                //input->samplePrint("input");
                //if (layer > 2) layers[layer-3]->act()->samplePrint("act");
                //if (layer > 3) layers[layer-4]->act()->samplePrint("act");
                fprop(input, 2, layers+layer, &_pt_hyper_params, state);
                error += bprop(input, tmp, 2, layers+layer, &_pt_hyper_params);

                //layers[layer+1]->act()->samplePrint("act");

                nInstance += _bp_hyper_params.batch_size;
                while (nEpoch*iperEpoch <= nInstance) {
                    nEpoch++;
                    _pt_hyper_params.learning_rate *= _pt_hyper_params.learning_rate_decay;
                    _pt_hyper_params.momentum += _pt_hyper_params.step_momentum;
                    if (_pt_hyper_params.momentum > _pt_hyper_params.max_momentum)
                        _pt_hyper_params.momentum = _pt_hyper_params.max_momentum;
                }
                lastCheck += _bp_hyper_params.batch_size;
                if (lastCheck >= _pt_hyper_params.check_interval) {
                    printf("\nLayer: %d\t Epoch: %d\tInstance: %d\tError: %f\n", layer, nEpoch, nInstance%iperEpoch, (float)(error/lastCheck));
                    LOG(printf("\nLayer: %d\t Epoch: %d\tInstance: %d\tError: %f\n", layer, nEpoch, nInstance%iperEpoch, (float)(error/lastCheck)));
                    lastCheck = 0;
                    error = 0.0;
                }
                delete x, y;
            }
            layers[layer]->scaleWeight(1-_pt_hyper_params.idrop_rate);
            delete tmp;
            delete layers[layer+1];
            delete layers;
        }
        _scaled_weight = true;
    }

    T fineTune(DData<T>* data, int total_epochs) {
        scaleWeight(false);
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank >= sgd_num_param_server)
#endif  
            data->start();
        int iperEpoch = data->instancesPerEpoch();
#ifdef DOWN_POUR_SGD
        if (mpi_world_rank < sgd_num_param_server)
           iperEpoch = data->totalInstancesPerEpoch(); 
#endif
        DMatrix<T> *x, *y;
        int nEpoch = 1;
        int nInstance = 0;
        T error = 0.0;
        bool checked = false;
        T lastError;
        int lastCheck = 0;
        int starting = time(0);
        
        CUDA_CALL(cudaThreadSynchronize());
        LOG(fprintf(flog, "Fine Tuning\n"));
        while ( nEpoch <= total_epochs ) {
#ifdef DOWN_POUR_SGD
            if (mpi_world_rank >= sgd_num_param_server)
#endif  
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
                _layers[_num_layers-1]->act()->samplePrint();
                y->samplePrint();
#ifdef ADMM
                printf("\nNode%d\tEpoch: %d\tInstance: %d\tError: %f\n", mpi_world_rank, nEpoch, nInstance%iperEpoch, (float)(error/lastCheck));
#elif defined(DOWN_POUR_SGD)
                if (mpi_world_rank == 0) {
                    printf("\nEpoch: %d\tInstance: %d\tError: %f\n", nEpoch, nInstance%iperEpoch, (float)(error/lastCheck));
                    LOG(fprintf(flog, "%f %d\n", (float)(error/lastCheck), time(0)-starting));
                }
#else
                printf("\nEpoch: %d\tInstance: %d\tError: %f\n", nEpoch, nInstance%iperEpoch, (float)(error/lastCheck));
                LOG(fprintf(flog, "%f %d\n", (float)(error/lastCheck), (int)time(0)-starting));
#endif
                checked = true;
                lastError = error/lastCheck;
                lastCheck = 0;
                error = 0.0;
            }
#ifdef DOWN_POUR_SGD
	    if (mpi_world_rank >= sgd_num_param_server)
#endif
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
