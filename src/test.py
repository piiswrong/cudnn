#!/usr/bin/env python
import os
import sys
import subprocess
import time

if False:
    tut_path = '/s0/jxie/TIMIT_tutorial/'
    log_path = '/s0/jxie/cudnn/log/'
else:
    tut_path = '/scratch/jxie/TIMIT_tutorial/'
    log_path = '/projects/grail/jxie/cudnn/log/'

def test(name, exp, start, ntotal):
    os.chdir(tut_path)
    fout = open(log_path+'%s_%d.acc'%(name,exp),'w')
    head = '%s_%d_'%(name,exp)
    tail = '.param'
    fdmlp = open(tut_path+'PARAMS/dmlp.nn', 'w')
    fdmlp.write(\
"""#include "../PARAMS/commonParams"

1

0
phoneDMLP          % Deep NN name
351                % # inputs = 11 frames * 39 MFCC features
TOTAL_NUM_STATES   % # outputs (output is P(State | MFCC frames) )
matrices:""")
    fhyper = open(log_path+'%s_%d.hyper'%(name,exp))
    hypers = {}
    for l in fhyper:
        l = l.strip().split('=')
        hypers[l[0]] = l[1]
    fhyper.close()
    num_layers = int(hypers['num_layers'])
    fdmlp.write('%d'%num_layers)
    for i in xrange(num_layers-1):
        fdmlp.write(' '+hypers['hidden_dim'])
    fdmlp.write(' TOTAL_NUM_STATES\n')
    for i in xrange(num_layers):
        neuron = ''
        if i == num_layers - 1:
            neuron = 'softmax'
        elif hypers['neuron'] == 'ReLU':
            neuron = 'rectlin'
        elif hypers['neuron'] == 'Oddroot':
            neuron = 'oddroot'
        else:
            print 'unsupported neuron type:', hypers['neuron']
        fdmlp.write('matrix%d:g%d\nsquash%d:%s\n'%(i,i,i,neuron))
    fdmlp.write('END')
    fdmlp.close()

    for xx in xrange(start, start+ntotal, 10):
        param_name = head+str(xx)+tail
        while not os.path.exists(log_path+param_name):
            print 'waiting for '+param_name
            time.sleep(5)
        last_size = os.stat(log_path+param_name).st_size
        while True:
            time.sleep(5)
            size = os.stat(log_path+param_name).st_size
            if size == last_size:
                break
            last_size = size
            print 'waiting for write to complete'
            
        os.system('cp %s%s %slearned_dmlp'%(log_path, param_name, tut_path))
        os.system('./dmlpvitcommand')
        #res = subprocess.check_output(['./dmlpscorecommand'])
        res = subprocess.Popen(['./dmlpscorecommand'], stdout=subprocess.PIPE).communicate()[0]
        print res
        acc = 0
        for l in res.strip().split('\n'):
            if l.startswith('WORD:'):
                for w in l.strip().split(' '):
                    if w.startswith('Acc='):
                        acc = float(w.strip()[4:])
                        break
                break
        fout.write('%d %f\n'%(xx, acc))
        fout.flush()
    fout.close()


test(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
