# this script takes as parameter the name of a user
processLine(){
  param="$@" # get all args
  ./hassolved.sh $param $CHALL 
}


if [ -z "$1" ]; then 
  echo "Please give a challenge to look upon on hacker.org" 
  exit 0
fi

CHALL="$1"
FILE="watched.cfg" 
if [ ! -f $FILE ]; then
  	echo "$FILE : does not exists, please launch discochal.sh first with some username"
  	exit 1
elif [ ! -r $FILE ]; then
  	echo "$FILE: can not read"
  	exit 2
fi

 
echo "Challenge $1 solved by" 
BAKIFS=$IFS
IFS=$(echo -en "\n\b")

exec 3<&0
exec 0<"$FILE"
while read -r line
do
	processLine $line
done
exec 0<&3
IFS=$BAKIFS
