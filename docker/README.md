:exclamation: The separation between build and run environments is not done yet. It should rely on the building of python wheels (cf. issue #2).

The two repositories have docker files that are used to generate build and run environments for MeshTools.

Log into the compass docker regisry:

```shell
docker login registry.gitlab.inria.fr
```

then in the `build` directory:

```shell
docker build -t registry.gitlab.inria.fr/charms/meshtools/build-environment .
docker push registry.gitlab.inria.fr/charms/meshtools/build-environment 
```

and in the `run` directory:

```shell
docker build -t registry.gitlab.inria.fr/charms/meshtools/run-environment .
docker push registry.gitlab.inria.fr/charms/meshtools/run-environment 
```
