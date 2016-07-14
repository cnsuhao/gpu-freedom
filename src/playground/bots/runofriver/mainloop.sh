#!/bin/bash
for (( ; ;  ))
do
   ./runofriver_collector.sh

   r=$(( $RANDOM % 50 ));
   TIMEOUT=1750
   let "TIMEOUT = $TIMEOUT + $r"
   echo "Random numer is $r"
   echo "Sleeping $TIMEOUT seconds..."
   sleep $TIMEOUT
done
