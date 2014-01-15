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

exps = [0, 2, 4, 6, 8]
exp_name = 'oddroot'

if len(sys.argv) <= 1:
    nodes = []
    for i in xrange(1,12):
        res = subprocess.Popen(['ssh', 'n%02d'%i, 'nvidia-smi'], stdout=subprocess.PIPE).communicate()[0]
        print res
        res = res.strip().split('\n')
        if filter(None, res[8].strip().split(' '))[-3] == '0%':
            nodes.append((i, 0))
        if filter(None, res[11].strip().split(' '))[-3] == '0%':
            nodes.append((i, 1))
    print nodes

    nodes2 = []
    for i in xrange(36,60):
        res = subprocess.Popen(['ssh', 'n%02d'%i, "ps -U jxie -f | grep 'test.py'"], stdout=subprocess.PIPE).communicate()[0]
        if len(res.strip().split('\n')) < 3:
            nodes2.append(i)
    print nodes2

    exit()

    for k in xrange(len(exps)):
        i,j = nodes[k]
        n = exps[k]
        cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/main -r 400 -d %d %s_%d >> /projects/grail/jxie/cudnn/log/%s_%d.o &'%(j, exp_name, n, exp_name, n)
        print cmd
        os.system("ssh n%02d '%s'"%(i,cmd))
        cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/test.py %s %d 400 600 > /projects/grail/jxie/cudnn/log/test_%s_%d.o &'%(exp_name, n, exp_name, n)
        print cmd
        os.system("ssh n%02d '%s'"%(nodes2[k], cmd))
        time.sleep(20)
else:
    for i in xrange(1, 14):
        cmd = "ssh n%02d '%s'"%(i,sys.argv[1])
        print cmd
        os.system(cmd)
