#!/bin/bash
for (( ; ;  ))
do
   ./runofriver_collector.sh
   cp ./data.csv ../../../../../  
   r=$(( $RANDOM % 50 ));
   TIMEOUT=1750
   let "TIMEOUT = $TIMEOUT + $r"
   echo "Random numer is $r"
   echo "Sleeping $TIMEOUT seconds..."
   sleep $TIMEOUT
done
