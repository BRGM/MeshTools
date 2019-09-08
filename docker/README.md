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

and in the `doc` directory:

```shell
docker build -t registry.gitlab.inria.fr/charms/meshtools/doc-environment .
docker push registry.gitlab.inria.fr/charms/meshtools/doc-environment 
```
