#!/bin/bash
#set -x
nvidia-smi
touch result.txt

src=~/data/
m=1000
n=1000
p=100
while [ $p -le 1000 ]; do
   echo $p
   file=${src}sensor$p.txt
   ./test $file $p 3 based $m $n>> result.txt
   p=`expr $p + 100`
done
exit 0

