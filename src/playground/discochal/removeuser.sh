cat watched.cfg | grep -v $1 > watched.new
mv watched.new watched.cfg

if [ "$2" != "nooutput"  ] 
then
echo "User $1 removed from watchlist!"
fi
