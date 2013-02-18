# this script simply downloads solvers for all levels
mkdir solvers &>/dev/null
source config.ini
 
for (( i=1; i<=$MAX_CHALLS; i++ ))
do
 echo "Downloading solvers for challenge $i"
 wget -O "solvers/solvers_$i.txt" "http://www.hacker.org/challenge/solvers.php?id=$i" #&> /dev/null

echo "Sleeping $TIMEOUT seconds..."
sleep $TIMEOUT
done
