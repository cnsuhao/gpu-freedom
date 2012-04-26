# this script simply downloads solvers for all levels
for i in {1..303}
do
 echo "Downloading solvers for challenge $i"
 wget -O "solvers/solvers_$i.txt" "http://www.hacker.org/challenge/solvers.php?id=$i" #&> /dev/null

echo "Sleeping 60 seconds..."
sleep 60
done
