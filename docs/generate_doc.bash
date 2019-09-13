#!/bin/bash

/bin/bash sdk/install_wheel.bash $1

# documentation generation
pushd public

# test: generate a landing page where wheels can be downloaded
python3 ../docs/generate_landing_page.py wheels

# Steps to generate sphinx doc
mkdir -p sphinx/reference
sphinx-apidoc ../MeshTools -o sphinx/reference
sphinx-build ../docs sphinx

popd
