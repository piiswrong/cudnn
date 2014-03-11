#!/usr/bin/env python
import os
import sys
import time
import subprocess




exps = list(xrange(0, 6))
exp_name = 'oddrootfinal'
test_only = False
resuming = -1

if len(sys.argv) <= 1:
    nodes = []
    if not test_only:
        for i in xrange(1,12):
            res = subprocess.Popen(['ssh', 'n%02d'%i, 'nvidia-smi'], stdout=subprocess.PIPE).communicate()[0]
            print res
            res = res.strip().split('\n')
            if filter(None, res[8].strip().split(' '))[-3] == '0%':
                nodes.append((i, 0))
            if filter(None, res[11].strip().split(' '))[-3] == '0%':
                nodes.append((i, 1))
    print nodes, len(nodes)

    nodes2 = []
    for i in xrange(1, 12):
        for dev in [0, 1]:
            res = subprocess.Popen(['ssh', 'n%02d'%i, "ps -U jxie -f | grep 'test.py %d'"%dev], stdout=subprocess.PIPE).communicate()[0]
            if len(res.strip().split('\n')) < 3:
                print i,dev
                nodes2.append((i,dev))
    print nodes2, len(nodes2)

    if (not test_only) and len(nodes) < len(exps):
        print 'not enough nodes!\n'
        exit()

    print 'confirm?'
    raw_input()

    for k in xrange(len(exps)):
        n = exps[k]
        if not test_only:
            i,j = nodes[k]
            if resuming >= 0:
                cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/main -d %d -r %d %s_%d > /projects/grail/jxie/cudnn/log/%s_%d.o &'%(j, resuming, exp_name, n, exp_name, n)
            else:
                cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/main -d %d %s_%d > /projects/grail/jxie/cudnn/log/%s_%d.o &'%(j, exp_name, n, exp_name, n)
            print cmd
            os.system("ssh n%02d '%s'"%(i,cmd))
        i, dev = nodes2[k]
        scriptfile = '/projects/grail/jxie/cudnn/src/test.py'
        outfile = "/projects/grail/jxie/cudnn/log/test_%s_%d.o"%(exp_name, n)
        cmd = "source ~/.profile; nohup python %s %d %s %d 0 1000 > %s &"%(scriptfile, dev, exp_name, n, outfile)
        print cmd
        os.system("ssh n%02d '%s'"%(i,cmd))
        #time.sleep(20)
else:
    for i in xrange(int(sys.argv[1]), int(sys.argv[2])+1):
        cmd = "ssh n%02d '%s'"%(i,sys.argv[3])
        print cmd
        os.system(cmd)
