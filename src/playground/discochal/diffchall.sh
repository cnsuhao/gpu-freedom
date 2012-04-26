if [ ! -f "users/$1.txt" ] 
then
   echo "Please run first ./discochal.sh $1 as we did not collect info on this user..."
   exit 0
fi
if [ ! -f "users/$2.txt" ] 
then
   echo "Please run first ./discochal.sh $2 as we did not collect info on this user..."
   exit 0
fi

diff "users/$1.txt" "users/$2.txt" | grep "^<"