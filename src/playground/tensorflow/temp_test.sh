#!/bin/bash
echo "initial temperature:"
./temp.sh
for j in `seq 1 10`;
do
   for i in `seq 1 10`;
   do
	./hello.py

   done

   echo $j
   ./temp.sh
done
