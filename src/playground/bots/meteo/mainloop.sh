#!/bin/bash
for (( ; ;  ))
do
   ./meteocollector_meteoswiss.sh
   r=$(( $RANDOM % 50 ));
   TIMEOUT=550
   let "TIMEOUT = $TIMEOUT + $r"
   echo "Random numer is $r"
   echo "Sleeping $TIMEOUT seconds..."
   sleep $TIMEOUT
done
