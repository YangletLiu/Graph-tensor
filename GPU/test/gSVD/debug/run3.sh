#!/bin/bash
#set -x
nvidia-smi
touch result.txt

src=~/data/
m=500
n=500
p=1000
while [ $p -le 2000 ]; do
   echo $p
   #file=${src}sensor$p.txt
   ./test $p 3 batched $m $n>> result.txt
   p=`expr $p + 100`
done
exit 0

