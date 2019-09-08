#!/bin/bash

WHEEL_TAG=$1

# FIXME: we'd better use a temporary file provided by the system
TMP_BUILD_DIR=build/tmp
mkdir -p $TMP_BUILD_DIR
# generate pip install command
PIP_EXE=`which pip3`
PIP_CMD_FILE=$TMP_BUILD_DIR/pip_install
python3 maintenance/get_pip_install.py $PIP_EXE MeshTools $WHEEL_TAG public/wheels > $PIP_CMD_FILE
cat $PIP_CMD_FILE
source $PIP_CMD_FILE
rm -rf $TMP_BUILD_DIR

# work in public directory to generate documentation
pushd public
# test: generate a landing page where wheels can be downloaded
python3 ../docs/generate_landing_page.py wheels

# Steps to generate sphinx doc
# sphinx-apidoc...
# sphinx-build....
popd
