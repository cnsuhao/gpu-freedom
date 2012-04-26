# this script takes as parameter the name of a user
if [ -z "$1" ]; then 
  echo "Please give a username to look upon on hacker.org" 
  echo "output will be stored on username.txt, containing the challenges the user solved"  
  exit 0
fi

source config.ini 
rm "users/$1.txt" &> /dev/null
echo "Challenges solved by $1" > "users/$1.txt"
for (( i=1; i<=$MAX_CHALLS; i++ ))
do
if [ -f "solvers/solvers_$i.txt" ]
then
	rm "user_$i.txt" &> /dev/null
	echo "Processing challenge $i looking for $1"
	cat "solvers/solvers_$i.txt" | grep -i $1 > "user_$1.txt"
 
	./removeif0.sh "user_$1.txt"
	if [ -f "user_$1.txt" ]
	then
		cat "solvers/solvers_$i.txt" | grep "<h2>Challenge" > "chal_$1.txt"
		cat "chal_$1.txt" >> "users/$1.txt"
	fi
fi
done

solved=$(wc -l "users/$1.txt" | awk '{print $1}')
let "solved-=1";
echo "$solved challenges solved by $1" >> "users/$1.txt"
cat "users/$1.txt"

echo "$1" >> watched.cfg

rm "user_$1.txt" &> /dev/null
rm "chal_$1.txt" &> /dev/null

