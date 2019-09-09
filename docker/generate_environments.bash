#!/bin/bash

REF_SLUG=${1:+":"}${1:""}

# order is important as doc depends on run 
for evt in build run doc
do
    echo "Building ${evt}-environment${REF_SLUG}"
    docker build -t registry.gitlab.inria.fr/charms/meshtools/${evt}-environment${REF_SLUG} ${evt}
    docker push registry.gitlab.inria.fr/charms/meshtools/${evt}-environment${REF_SLUG}
done
