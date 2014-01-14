#!/usr/bin/env python
import os
import sys
import time


nodes = [3, 4, 4, 5, 5, 6, 6, 8, 8]
nodes = xrange(36,46)
nodes = [1, 2, 3, 4, 5]
nodes = [2,4]+list(xrange(6,12))
nodes = nodes + nodes

nodes = xrange(36,60)

if len(sys.argv) <= 1:
    n = 0
    for i in nodes:
        #cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/main -r 400 oddroot_%d >> /projects/grail/jxie/cudnn/log/oddroot_%d.o &'%(n,n)
        cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/test.py oddroot %d 400 600 > /projects/grail/jxie/cudnn/log/test_%d.o &'%(n,n)
        print cmd
        os.system("ssh n%02d '%s'"%(i,cmd))#sys.argv[1]))
        n += 2
        if n > 8:
            break
        #time.sleep(20)
else:
    for i in xrange(1, 14):
        cmd = "ssh n%02d '%s'"%(i,sys.argv[1])
        print cmd
        os.system(cmd)
