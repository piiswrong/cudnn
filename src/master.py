import math
import shutil
from sets import Set
import os
import matplotlib.pyplot as plt
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

log_path = '../log/'


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
    for neuron in [ 'Logistic', 'Oddroot', 'ReLU']:
        for i in [ 5, 10, 20 ]:
            for ntotal_param in [ 1e7 , 2e7]:
                net['num_layers'] = i
                net['neuron'] = neuron
                net['bp_epochs'] = 1000
                t = (-501+math.sqrt((501.0+i)**2+4.0*(i-2)*ntotal_param))/(2.0*(i-2))
                net['hidden_dim'] = int((t+8)/16)*16 - 1

                bpHyper['learning_rate'] = 0.1

                bpHyper['hdrop_rate'] = 0.2
                
                fout = open('%s%s_%d.hyper'%(log_path,exp_name, id), 'w')
                writeExp(fout, net, norder, ptHyper, bpHyper, horder)

                if neuron != 'ReLU':
                    pre = '%s%s_d%d_w%d.pre'%(log_path, neuron,i,net['hidden_dim'])
                    param = '%s%s_%d.param'%(log_path,exp_name,id)
                    #shutil.copy2(pre, param)
                    fpre = open(pre)
                    lines = fpre.readlines()
                    fpre.close()
                    fparam = open(param, 'w')
                    fparam.writelines(lines[2:])
                    fparam.close()
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




    hyper_name = []
    for i in diff:
        s = params[0][i].strip().split('=')[0]
        if i >= nnet and i < nnet+nhyper:
            s = 'pt_'+s
        hyper_name.append(s)
        print s+'\t',
    print '\n'
    
    
    hyper_value = []
    acc_list = []
    for exp,param,id in zip(exps, params, xrange(len(exps))):
        print '%d:'%id
        hyper_value.append([])
        for i in diff:
            v = param[i].strip().split('=')[1]
            hyper_value[-1].append(v)
            print str(v)+'\t',

        fin = open('%s%s_%d.log'%(log_path, exp_name, exp))
        lines = fin.readlines()
        fin.close()
        last_acc = -1
        train_acc = {}
        j = 0
        for i in xrange(len(lines)):
            if lines[i].startswith('Fine') and last_acc != -1:
                train_acc[j] = last_acc
                j += 10
            if lines[i].startswith('0.'):
                last_acc = float(lines[i].strip().split(' ')[0])
        train_acc[j] = last_acc


        fin = open('%s%s_%d.acc'%(log_path, exp_name, exp))
        lines = fin.readlines()
        lines = [ (int(l.strip().split(' ')[0]),float(l.strip().split(' ')[1])) for l in lines ]
        fin.close()
        maxacc = 0.0
        print 'acc per 10 epochs'
        acc_list.append([])
        for e,acc in sorted(lines, key=lambda x: x[0]):
            if acc > maxacc:
                maxacc = acc
            try:
                te = "%.2f"%train_acc[e]
            except:
                te = "N/A"
            acc_list[-1].append(acc)
            print '%.2f(%s)'%(acc, te) + '\t',
            i += 1
        print 'max = %.2f'%maxacc+'\n'

    if not os.path.exists(log_path+exp_name+'_plots'):
        os.mkdir(log_path+exp_name+'_plots')
    for i in xrange(len(hyper_name)):
        groups = Set()
        for j in xrange(len(hyper_value)):
            groups.add(hyper_value[j][i])
        for v in groups:
            plt.hold(True)
            for j in xrange(len(hyper_value)):
                if hyper_value[j][i] != v:
                    continue
                label = ''
                for k in xrange(len(hyper_name)):
                    if k == i:
                        continue
                    label += hyper_name[k] + '=' + str(hyper_value[j][k]) + ' '
                plt.plot(xrange(10, len(acc_list[j])*10+1, 10), acc_list[j], label=label)
            plt.title(hyper_name[i] + '=' + str(v))
            plt.xlabel('number of epochs')
            plt.ylabel('accuracy')
            plt.legend(loc = 'lower right')
            plt.savefig(log_path+exp_name+'_plots/'+hyper_name[i] + '-' + str(v) + '.png')
            plt.clf()
            

#makeReport('oddroot', [0,2,4,6,8])
#makeReport('ReLU', xrange(0,15))
#makeReport('ReLUdropout', xrange(0,18))
#makeExp('ReLUdropout', 1e7)
#makeExp('oddrootnew', 1e7)
#makeReport('oddrootnew', xrange(5))
#makeExp('ReLU', 1e7)
#makeExp('sigmoid', 1e7)
#makeReport('sigmoid', xrange(0,15))
#makeExp('oddrootnew1', 1e7)
#makeExp('all1', [1e7, 2e7])
#makeExp('all2', [2e7])
#makeReport('all2', xrange(0,18))
#makeExp('oddrootresume', [])
#makeReport('oddrootresume', xrange(0,6))
#makeExp('all3', [] )
makeReport('all3', xrange(0, 18))




