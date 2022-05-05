This directory contains yaml files to set
[conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments).
To use them just run:

```shell
conda env create -f file.yml [-n my_fancy_name]
```

There are two flawors:
- `meshtools.yml` defines an environment where you can compile meshtools
  it can serve for development purposes
  (you will have first to git clone the MeshTools repository)
- `meshtools-latest.yml` is the same environment as above but it will also
  install the latest version of meshtools available on github so that
  you will be able to readily use the latest version of MeshTools an run
  a script containing `import MeshTools`
