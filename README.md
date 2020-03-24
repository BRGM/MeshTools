
The documentation is currently served on
[INRIA's gitlab server pages](https://charms.gitlabpages.inria.fr/MeshTools).


The MeshTools package has originally been developed to share a common mesh data
structure with the ComPASS multiphase multicomponent flow simulation code
(cf. the [CHARMS project website](http://www.anr-charms.org)).

For the moment MeshTools essentially serves as a pre and post-processing module
for the ComPASS code. Yet, there is no plan to make it an exhaustive tool for
mesh management and or generic conversion operations. If you look for such
packages you may also consider alternative projects[^altmesh] such as
[RingMesh](https://github.com/ringmesh/RINGMesh),
[meshio](https://pypi.org/project/meshio/), the
[ Open-Asset-Importer-Library](http://www.assimp.org/) and its
[python bindings](https://pypi.org/project/pyassimp/)
that you can also find in
[asssimp's main code repository](https://github.com/assimp/assimp)...

[^altmesh]: any other reference is welcome here...

There is still no documentation but you may start with the
[installation instructions](INSTALL.md).

:exclamation: Beware that from commit 3571466f342bd8e048fbc37368225b1ae7d14b6b
you will need a patched version of CGAL 4.12.1
(cf. https://github.com/CGAL/cgal/pull/3377).
