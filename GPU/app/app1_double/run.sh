#!/bin/bash
#set -x
nvidia-smi
touch result.txt

src=./CPark360_20.txt
#P=1
frame=10
while [ $frame -le 20 ]; 
do
    for k in $( seq 1 5 )
do
    	echo $frame
    	file=U$frame.txt
    	./app_frame_recovery $frame $src $file>> result.txt
done
     frame=`expr $frame + 1`;
done
exit 0

