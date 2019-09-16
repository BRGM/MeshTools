#!/bin/bash

/bin/bash sdk/install_wheel.bash $1 wheel

pushd docs
for f in README INSTALL LICENSE
do
    cp -vf ../$f.md .
done
# FIXME: inconsitenscy between directory names
ln -s ../wheel wheels
python3 generate_wheels_page.py wheel
sphinx-apidoc ../MeshTools -o reference
sphinx-build . html
cp -rf html/* ../public/
popd
