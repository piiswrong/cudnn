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
             'hdrop_out':0,
             'hdrop_rate':0.5,
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
             'neuron':'Logistic',
             'pt_epochs':0.2,
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
    for i in xrange(10, 20, 2):
        for rate in [0.0001, 0.001]:
            net['num_layers'] = i
            t = (-501+math.sqrt((501.0+i)**2+4.0*(i-2)*ntotal_param))/(2.0*(i-2))
            net['hidden_dim'] = int((t+8)/16)*16 - 1
            
            fout = open('/homes/grail/jxie/cudnn/log/%s_%d.hyper'%(exp_name, id), 'w')
            writeExp(fout, net, norder, ptHyper, bpHyper, horder)
            id += 1

makeExp('test', 1e7)












