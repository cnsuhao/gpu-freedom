#/bin/bash
FILESIZE=$(stat -c%s "$1")
if [ "$FILESIZE" -eq "0" ]; then
  rm $1
fi
