import math
"""
fout = open('test.hyper', 'w')


fout.write('num_layers=' + str(20) + '\n')
fout.write('hidden_dim=' + str(647) + '\n')
fout.write('Logistic\n')
fout.write('pt_epochs=' + str(0.2) + '\n')
fout.write('bp_epochs=' + str(200) + '\n')
fout.write('learning_rate=' + str(0.1) + '\n')
fout.write('learning_rate_decay=' + str(0.998) + '\n')
fout.write('momentum=' + str(0.5) + '\n')
fout.write('max_momentum=' + str(0.9) + '\n')
fout.write('step_momentum=' + str(0.04) + '\n')
fout.write('weight_decay=' + str(0) + '\n')
fout.write('decay_rate=' + str(1e-6) + '\n')
fout.write('idrop_out=' + str(0) + '\n')
fout.write('idrop_rate=' + str(0.2) + '\n')
fout.write('hdrop_out=' + str(1) + '\n')
fout.write('hdrop_rate=' + str(0.5) + '\n')
fout.write('batch_size=' + str(128) + '\n')
fout.write('check_interval=' + str(10000) + '\n')
"""

log_path = '/projects/grail/jxie/cudnn/log/'


def makeHyper():
    return { 'learning_rate':0.1,
             'learning_rate_decay':0.998,
             'momentum':0.5,
             'max_momentum':0.9,
             'step_momentum':0.04,
             'weight_decay':0,
             'decay_rate':1e-6,
             'idrop_out':0,
             'idrop_rate':0.2,
             'hdrop_out':1,
             'hdrop_rate':0.2,
             'batch_size':128,
             'check_interval':10000,
             }, ['learning_rate',
             'learning_rate_decay',
             'momentum',
             'max_momentum',
             'step_momentum',
             'weight_decay',
             'decay_rate',
             'idrop_out',
             'idrop_rate',
             'hdrop_out',
             'hdrop_rate',
             'batch_size',
             'check_interval']

def makeNet():
    return { 'num_layers':5,
             'hidden_dim':2000,
             'neuron':'Oddroot',
             'pt_epochs':0.0,
             'bp_epochs':200
            }, ['num_layers',
             'hidden_dim',
             'neuron',
             'pt_epochs',
             'bp_epochs']

def writeDic(fout, dic, order):
    for s in order:
        fout.write(s+'='+str(dic[s])+'\n')

def writeExp(fout, net, norder, ptHyper, bpHyper, horder):
    writeDic(fout, net, norder)
    writeDic(fout, ptHyper, horder)
    writeDic(fout, bpHyper, horder)

def makeExp(exp_name, ntotal_param):
    net, norder = makeNet()
    ptHyper, horder = makeHyper()
    bpHyper, horder = makeHyper()
    id = 0
    i = 8
    for i in xrange(11, 20, 3):
        for rate in [0.1, 0.5]:
            for drop in [0.00, 0.02, 0.08]:
                net['num_layers'] = i
                net['neuron'] = 'ReLU'
                net['bp_epochs'] = 100
                t = (-501+math.sqrt((501.0+i)**2+4.0*(i-2)*ntotal_param))/(2.0*(i-2))
                net['hidden_dim'] = int((t+8)/16)*16 - 1

                bpHyper['learning_rate'] = rate
                bpHyper['hdrop_rate'] = drop
                
                fout = open('%s%s_%d.hyper'%(log_path,exp_name, id), 'w')
                writeExp(fout, net, norder, ptHyper, bpHyper, horder)
                id += 1


def makeReport(exp_name, exps):
    nnet = len(makeNet()[0])
    nhyper = len(makeHyper()[0])
    params = []
    for exp in exps:
        fin = open('%s%s_%d.hyper'%(log_path, exp_name, exp))
        params.append(fin.readlines())
        fin.close()
    diff = []
    for i in xrange(nnet+2*nhyper):
        for param in params[1:]:
            if param[i] != params[0][i]:
                diff.append(i)
                break

    print 'Global hyper-parameters:'
    for i in xrange(0,nnet):
        if not i in diff:
            print params[0][i].strip()

    print '\npre-training hyper-parameters:'
    for i in xrange(nnet,nnet+nhyper):
        if not i in diff:
            print params[0][i].strip()
            
    print '\nfine tuning hyper-parameters:'
    for i in xrange(nnet+nhyper, nnet+2*nhyper):
        if not i in diff:
            print params[0][i].strip()




    for i in diff:
        s = params[0][i].strip().split('=')[0]
        if i >= nnet and i < nnet+nhyper:
            s = 'pt_'+s
        print s+'\t',
    print '\n'
    for exp,param in zip(exps, params):
        for i in diff:
            v = param[i].strip().split('=')[1]
            print str(v)+'\t',
        fin = open('%s%s_%d.acc'%(log_path, exp_name, exp))
        lines = [ (int(l.strip().split(' ')[0]),float(l.strip().split(' ')[1])) for l in fin ]
        fin.close()
        maxacc = 0.0
        print 'acc per 10 epochs'
        for e,acc in sorted(lines, key=lambda x: x[0]):
            if acc > maxacc:
                maxacc = acc
            print '%.2f'%acc + '\t',
        print 'max = %.2f'%maxacc+'\n'

#makeReport('oddroot', [0,2,4,6,8])
#makeReport('ReLU', xrange(0,15))
makeReport('ReLUdropout', xrange(0,18))
#makeExp('ReLUdropout', 1e7)




