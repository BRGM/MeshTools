FROM registry.gitlab.inria.fr/charms/meshtools/build-environment:latest AS builder
WORKDIR /source
COPY ./ /source/MeshTools
RUN cd ./MeshTools \
 && CGAL_DIR=/build/thirdparties/cgal python3 setup.py bdist_wheel

FROM registry.gitlab.inria.fr/charms/meshtools/run-environment:latest
WORKDIR /wheels
COPY --from=builder /source/MeshTools/dist/MeshTools-0.0.1-cp35-cp35m-linux_x86_64.whl .
RUN pip3 install MeshTools-0.0.1-cp35-cp35m-linux_x86_64.whl

WORKDIR /data
ENTRYPOINT ["/bin/bash"]
