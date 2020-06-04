#!/bin/bash
#set -x
nvidia-smi
touch result.txt

p=1000
while [ $p -le 20000 ]; do
   echo $p 
   ./test $p >> result.txt
   p=`expr $p + 1000`
done
exit 0

