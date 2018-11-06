# check docker version if label builder is not accepted
From registry.gitlab.inria.fr/charms/meshtools/build-environment:latest as builder
WORKDIR /source
COPY ./ /source/MeshTools
RUN cd ./MeshTools && python3 setup.py bdist_wheel

From registry.gitlab.inria.fr/charms/meshtools/run-environment:latest
WORKDIR /wheels/
COPY --from=builder /source/MeshTools/dist/MeshTools-0.0.1-cp36-cp36m-linux_x86_64.whl .
RUN pip3 install MeshTools-0.0.1-cp36-cp36m-linux_x86_64.whl

WORKDIR /data/
ENTRYPOINT ["/bin/bash"]
