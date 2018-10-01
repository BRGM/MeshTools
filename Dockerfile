# check docker version if label builder is not accepted
From registry.gitlab.inria.fr/charms/meshtools/build-environment:latest as builder
WORKDIR /source
COPY ./ /source/MeshTools
RUN pip3 install ./MeshTools

#From registry.gitlab.inria.fr/charms/compass/run-environment:latest
#ENV PYTHONPATH=/build/meshtoolsModule:/build/compassModule
#WORKDIR /build/
##COPY ./docker/script/docker_entrypoint.sh /
#COPY --from=builder /source/ComPASS-develop/python ./compassModule
#COPY --from=builder /source/ComPASS-develop/thirdparty/meshtools/python ./meshtoolsModule
#VOLUME [/data]

#WORKDIR /data
#
#RUN useradd --create-home -s /bin/bash compass && chown compass:compass /data
#USER compass
#
##to uncomment when bash script will be done.
##ENTRYPOINT ["/bin/bash","/docker_entrypoint.sh"]
#ENTRYPOINT ["python3"]
##ENTRYPOINT ["mpirun", "-n", "`nproc`", "python3"]

ENTRYPOINT ["/bin/bash"]

