cd gpu_freedom
git pull
cd ..
cp -r gpu_freedom gpu_tarball
cd gpu_tarball
cd src/server/conf
rm config.inc.php
cd ../../..
rm -rf .git
cd ..
tar -cf gpu_tarball.tar ./gpu_tarball
gzip gpu_tarball.tar
rm -rf ./gpu_tarball
chmod 755 gpu_tarball.tar.gz
