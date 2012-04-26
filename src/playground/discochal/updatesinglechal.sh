# this script simply downloads solvers for a particular challenge ID
echo "Downloading solvers for challenge $1"
wget -O "solvers/solvers_$i.txt" "http://www.hacker.org/challenge/solvers.php?id=$1"
echo "Done :-)"

