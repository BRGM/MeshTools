#!/bin/bash

/bin/bash sdk/install_wheel.bash $1 wheels

pushd docs
ln -s ../wheels
for f in README INSTALL LICENSE
do
    cp -vf ../$f.md .
done
python3 generate_wheels_page.py wheels
sphinx-apidoc ../MeshTools -o reference
sphinx-build . html
cp -rf html/* ../public/
cp -rfv wheels ../public/
popd
