#!/bin/sh
HOST=`hostname -s`
USER=`whoami`
#echo $1
touch ~/$HOST.$USER
while [ -e ~/$HOST.$USER ]; do
	sleep 30
done
