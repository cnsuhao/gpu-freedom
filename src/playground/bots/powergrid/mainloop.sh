#!/bin/bash
for (( i=1; i<=16; i++ ))
do
   ./frequencycollector_swissgrid.sh
   r=$(( $RANDOM % 1200 ));
   TIMEOUT=2000
   let "TIMEOUT = $TIMEOUT + $r"
   echo "Random numer is $r"
   echo "Sleeping $TIMEOUT seconds..."
   sleep $TIMEOUT
done
