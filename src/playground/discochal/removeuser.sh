if [ -f "watched.cfg" ]
then 
	cat watched.cfg | grep -v $1 > watched.new
	mv watched.new watched.cfg
fi

if [ "$2" != "nooutput"  ] 
then
echo "User $1 removed from watchlist!"
fi
