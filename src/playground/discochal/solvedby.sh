if [ ! -f "users/$1.txt" ] 
then
   echo "Please run first ./discochal.sh $1 as we did not collect info on this user..."
   exit 0
fi

cat "users/$1.txt"
