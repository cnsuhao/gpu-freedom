cat watched.cfg | grep -v $1 > watched.new
mv watched.new watched.cfg
echo "User $1 removed from watchlist!"
