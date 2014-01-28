#!/usr/bin/env python
import os
import sys
import time
import subprocess


nodes = [3, 4, 4, 5, 5, 6, 6, 8, 8]
nodes = xrange(36,46)
nodes = [1, 2, 3, 4, 5]
nodes = [2,4]+list(xrange(6,12))
nodes = nodes + nodes

nodes = xrange(36,60)

exps = list(xrange(0,15))
exp_name = 'ReLU'

if len(sys.argv) <= 1:
    nodes = []
    for i in xrange(1,12):
        if i == 6:
            continue
        res = subprocess.Popen(['ssh', 'n%02d'%i, 'nvidia-smi'], stdout=subprocess.PIPE).communicate()[0]
        print res
        res = res.strip().split('\n')
        if filter(None, res[8].strip().split(' '))[-3] == '0%':
            nodes.append((i, 0))
        if filter(None, res[11].strip().split(' '))[-3] == '0%':
            nodes.append((i, 1))
    print nodes

    nodes2 = []
    res = list(xrange(14,61))
    lines = subprocess.Popen(['qstat', '-f', '-q', 'all.q'], stdout=subprocess.PIPE).communicate()[0].strip().split('\n')
    lines.reverse()
    res = [ int(i[7:9]) for i in lines if i.startswith('all.q') and int(i[7:9]) in res and i.strip().split(' ')[-1] == 'lx26-amd64' ]
    for i in res:
        res = subprocess.Popen(['ssh', 'n%02d'%i, "ps -U jxie -f | grep 'test.py'"], stdout=subprocess.PIPE).communicate()[0]
        if len(res.strip().split('\n')) < 3:
            nodes2.append(i)

    
    print nodes2
    #nodes2 = list(xrange(1,15))

    if len(nodes) < len(exps) or len(nodes2) < len(exps):
        print 'not enough nodes!\n'
        exit()

    print 'confirm?'
    raw_input()

    for k in xrange(len(exps)):
        n = exps[k]
        i,j = nodes[k]
        cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/main -d %d %s_%d > /projects/grail/jxie/cudnn/log/%s_%d.o &'%(j, exp_name, n, exp_name, n)
        print cmd
        #os.system("ssh n%02d '%s'"%(i,cmd))
        #cmd = 'source ~/.profile; nohup python /projects/grail/jxie/cudnn/src/test.py %s %d 0 1000 > /projects/grail/jxie/cudnn/log/test_%s_%d.o &'%(exp_name, n, exp_name, n)

        unit = 15750/len(exps)
        s = k*unit
        if k == len(exps)-1:
            t = 15750
        else:
            t = (k+1)*unit

        cmd = 'source ~/.profile; nohup python /projects/grail/jxie/paris/download.py %d %d > /projects/grail/jxie/paris/%d.o &'%(s, t, k)
        cmd = "ssh n%02d '%s'"%(nodes2[k%len(nodes2)], cmd)
        print cmd
        os.system(cmd)
        #time.sleep(20)
else:
    for i in xrange(int(sys.argv[1]), int(sys.argv[2])+1):
        cmd = "ssh n%02d '%s'"%(i,sys.argv[3])
        print cmd
        os.system(cmd)
