# this script updates all users and should be executed after ./updatechals.sh
processLine(){
  param="$@" # get all args
  ./discochal.sh $param "nowatchupdate"
}

echo "Updating all users in watchlist"
FILE="watched.cfg" 
if [ ! -f $FILE ]; then
  	echo "$FILE : does not exists, please launch discochal.sh first with some username."
  	exit 1
elif [ ! -r $FILE ]; then
  	echo "$FILE: can not read"
  	exit 2
fi

 
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

echo "Done!"
