FROM alpine:latest

ARG MESHTOOLS_WHEEL_PATH
WORKDIR /wheel
COPY $MESHTOOLS_WHEEL_PATH .

# The mount point to retrieve the wheel
VOLUME /localfs
ENTRYPOINT [ "/bin/busybox" ]
CMD [ "cp", "-rv", "/wheel", "/localfs" ]
