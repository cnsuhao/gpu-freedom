#/bin/bash
FILESIZE=$(du -k "$1" | awk '{print $1}')

if [ "$FILESIZE" -eq "0" ]; then
  rm $1
fi
