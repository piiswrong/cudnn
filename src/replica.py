#!/usr/bin/env python
import os
import sys
import time
import subprocess




exps = list(xrange(3, 6))
exp_name = 'all2'
test_only = True 
resuming = 0 

if len(sys.argv) <= 1:
    nodes = []
    if not test_only:
        for i in xrange(1,12):
            if i == 6: 
                continue
            res = subprocess.Popen(['ssh', 'n%02d'%i, 'nvidia-smi'], stdout=subprocess.PIPE).communicate()[0]
            print res
            res = res.strip().split('\n')
            if i != 4 and filter(None, res[8].strip().split(' '))[-3] == '0%':
                nodes.append((i, 0))
            if filter(None, res[11].strip().split(' '))[-3] == '0%':
                nodes.append((i, 1))
    print nodes, len(nodes)

    """
    nodes2 = []
    res = list(xrange(14,61))
    res.reverse()
    lines = subprocess.Popen(['qstat', '-f', '-q', 'all.q'], stdout=subprocess.PIPE).communicate()[0].strip().split('\n')
    lines.reverse()
    res = [ int(i[7:9]) for i in lines if i.startswith('all.q') and int(i[7:9]) in res and i.strip().split(' ')[-1] == 'lx26-amd64' ]
    for i in res:
        res = subprocess.Popen(['ssh', 'n%02d'%i, "ps -U jxie -f | grep 'test.py'"], stdout=subprocess.PIPE).communicate()[0]
        if len(res.strip().split('\n')) < 3:
            print i
            nodes2.append(i)
            if len(nodes2) >= len(exps):
                break

    
    print nodes2, len(nodes2)
    #nodes2 = list(xrange(1,15))
    """

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
        scriptfile = '/projects/grail/jxie/cudnn/src/test.py'
        outfile = "/projects/grail/jxie/cudnn/log/test_%s_%d.o"%(exp_name, n)
        cmd = "qsub -S /usr/bin/python -V -q notcuda.q -pe orte 32 -j y -o %s %s %s %d 0 1000"%(outfile, scriptfile, exp_name, n)
        print cmd
        os.system(cmd)
        #time.sleep(20)
else:
    for i in xrange(int(sys.argv[1]), int(sys.argv[2])+1):
        cmd = "ssh n%02d '%s'"%(i,sys.argv[3])
        print cmd
        os.system(cmd)
