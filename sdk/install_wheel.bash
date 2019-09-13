#!/bin/bash

WHEEL_TAG=$1
WHEEL_DIRECTORY=${2:-"public/wheels"}

# FIXME: we'd better use a temporary file provided by the system
TMP_BUILD_DIR=build/tmp
mkdir -p $TMP_BUILD_DIR
# generate pip install command
PIP_EXE=`which pip3`
PIP_CMD_FILE=$TMP_BUILD_DIR/pip_install
python3 maintenance/get_pip_install.py $PIP_EXE MeshTools $WHEEL_TAG $WHEEL_DIRECTORY > $PIP_CMD_FILE
cat $PIP_CMD_FILE
source $PIP_CMD_FILE
rm -rf $TMP_BUILD_DIR
