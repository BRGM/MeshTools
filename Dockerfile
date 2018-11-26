FROM registry.gitlab.inria.fr/charms/meshtools/build-environment:latest AS builder
WORKDIR /source
COPY ./ /source/MeshTools
RUN cd ./MeshTools && python3.7 setup.py bdist_wheel

FROM registry.gitlab.inria.fr/charms/meshtools/run-environment:latest
WORKDIR /wheels
COPY --from=builder /source/MeshTools/dist/MeshTools-0.0.1-cp37-cp37m-linux_x86_64.whl .
RUN pip install MeshTools-0.0.1-cp37-cp37m-linux_x86_64.whl

WORKDIR /data
ENTRYPOINT ["/bin/bash"]
