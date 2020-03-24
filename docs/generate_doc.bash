#!/bin/bash

pushd docs
for f in README INSTALL LICENSE
do
    cp -vf ../$f.md .
done
sphinx-apidoc ../MeshTools -o reference
sphinx-build . html
cp -rf html/* ../public/
popd
