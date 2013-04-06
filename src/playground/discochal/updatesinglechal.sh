# this script simply downloads solvers for a particular challenge ID
function is_integer() {    
   s=$(echo $1 | tr -d 0-9)
   if [ -z "$s" ]; then        
      return 0    
   else        
      return 1    
   fi
}

if ! is_integer $1; then
  echo "You need to specify a challenge id as a number!"
  exit 0
fi
echo "Updating solvers for challenge $1"
mkdir solvers &>/dev/null
wget -O "solvers/solvers_$1.txt" "http://www.hacker.org/challenge/solvers.php?id=$1"
echo "Done :-)"

