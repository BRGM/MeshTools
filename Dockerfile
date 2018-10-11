# check docker version if label builder is not accepted
From registry.gitlab.inria.fr/charms/meshtools/build-environment:latest as builder
WORKDIR /source
COPY ./ /source/MeshTools
RUN cd ./MeshTools && python3 setup.py bdist_wheel

From registry.gitlab.inria.fr/charms/meshtools/run-environment:latest
WORKDIR /data/
COPY --from=builder /source/MeshTools/dist/MeshTools-*-cp36-cp36m-linux_x86_64.whl .
# we are assuming only one wheel was copied
RUN pip3 install *.whl
RUN rm *.whl

ENTRYPOINT ["/bin/bash"]

