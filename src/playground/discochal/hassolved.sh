rm "challenge.txt" &> /dev/null
if [ -f "users/$1.txt" ]
then
	cat "users/$1.txt" | grep -i $2 > challenge.txt
	./removeif0.sh challenge.txt
	if [ -f "challenge.txt" ]
	then
		echo "$1"
	fi
fi
rm "challenge.txt" &> /dev/null