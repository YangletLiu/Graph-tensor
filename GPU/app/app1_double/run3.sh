#!/bin/bash
#set -x
nvidia-smi
touch result.txt

src=./CPark360_20.txt
#P=1
frame=10
while [ $frame -le 10 ]; 
do
    for k in $( seq 1 20 )
do
    	echo $k
    	file=U$frame.txt
    	./app_frame_recovery $frame $src $file $k>> result.txt
done
     frame=`expr $frame + 1`;
done
exit 0

