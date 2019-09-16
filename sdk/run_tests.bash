#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "You must provide the pre-release wheel tag"
    exit -1
fi

/bin/bash sdk/install_wheel.bash $1 wheel

pytest tests/ci