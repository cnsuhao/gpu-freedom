#!/bin/bash
r=$(( $RANDOM % 70 ));
TIMEOUT=130
let "TIMEOUT = $TIMEOUT + $r"
echo "Random numer is $r"
echo "Sleeping $TIMEOUT seconds..."
sleep $TIMEOUT
