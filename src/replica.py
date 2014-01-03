#!/usr/bin/env python
import os
import sys
import time


if len(sys.argv) <= 1:
    n = 0
    for i in xrange(5,10):
        for j in xrange(2):
            cmd = 'source ~/.profile; nohup /projects/grail/jxie/cudnn/src/main ReLU_%d > /projects/grail/jxie/cudnn/log/ReLU_%d.o &'%(n,n)
            print cmd
            os.system("ssh n%02d '%s'"%(i,cmd))#sys.argv[1]))
            n += 1
            time.sleep(5)
else:
    for i in xrange(1,14):
        cmd = "ssh n%02d '%s'"%(i,sys.argv[1])
        print cmd
        os.system(cmd)
