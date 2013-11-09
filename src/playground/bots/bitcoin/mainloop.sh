#!/bin/bash
for (( ; ; ))
do
   python ./bot/main.py ticker
   r=$(( $RANDOM % 1200 ));
   TIMEOUT=2000
   let "TIMEOUT = $TIMEOUT + $r"
   echo "Random numer is $r"
   echo "Sleeping $TIMEOUT seconds..."
   sleep $TIMEOUT
done
