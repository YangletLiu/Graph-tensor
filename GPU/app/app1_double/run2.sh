#!/bin/bash
#set -x
nvidia-smi
touch result2.txt

src=./CPark360_20.txt
#P=1
frame=10
while [ $frame -le 10 ]; 
do
    for k in $( seq 4 4 )
do
        for j in $( seq 1 5 )

do
    	echo $k
    	file=U$frame.txt
    	./app_frame_recovery $frame $src $file $k>>result2.txt
done
done
     frame=`expr $frame + 1`;
done
exit 0

