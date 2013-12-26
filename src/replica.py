#! /usr/bin/python
import os
import sys

for i in xrange(1,14):
    print "ssh n%02d '%s'"%(i,sys.argv[1])
    os.system("ssh n%02d '%s'"%(i,sys.argv[1]))
