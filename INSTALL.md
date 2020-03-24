## Requirements

### Compilers

You will need a relatively recent C++-14 compiler.

Compilation has been successfully tested with:
  * gcc-6.3 and gcc-7.2 on Linux
  * Intel icpc-17 on Linux
  * Microsoft Visual Studio 2015 (i.e. version 14.0) on Windows

### Build system

You will need a recent version of [CMake](https://cmake.org/).
Version 3.4 or above should be ok.

The windows installer that you will find in
the [download section of the cmake site](https://cmake.org/download/)
will install a code with a graphical user interface.

On Linux you may want to use ccmake or cmake-gui. On ubuntu, these are
available through the package manager (e.g. `sudo apt-get install ccmake`).

## Dependencies

### External dependencies

MeshTools can optionally be built with CGAL extensions.

:exclamation: Beware that from commit 3571466f342bd8e048fbc37368225b1ae7d14b6b
you will need a version of CGAL that integrates the following
[patch](https://github.com/CGAL/cgal/pull/3377).

On Linux you may just install the ad-hoc package using your favourite package
manager.
As from version 4.12, CGAL can be used in a header only mode you may also just
want to download the [CGAL source code](https://github.com/CGAL/cgal/releases)
and set the CMake `CGAL_DIR` variable point to the source directory.

If you install MeshTools using pip (cf. below) and want to use a specific
version of CGAL  don't forget to use the `MESHTOOLS_WITH_CGAL_DIR` environment
variable to pass the value of `CGAL_DIR` to cmake.

On Windows you may need precompiled versions of the *gmp* and *mpfr* libraries
that you may obtain from the
[Geometry Factory website](https://doc.cgal.org/latest/Manual/installation.html).

On Windows, you can use the CGAL Installer from the
[github CGAL page](https://github.com/CGAL/cgal/releases) that will install
precompiled versions of the gmp and mpfr libraries. Then just define the
CGAL_DIR environment variable to where the CGAL source has been installed or
use the MESHTOOLS_WITH_CGAL_DIR environment variable to pass the value of
CGAL_DIR to cmake (if for any, obviously good, reason you do not want to use
CGAL_DIR).

CGAL brings in a dependency to a few [boost](https://www.boost.org/) libraries.

### Packaged dependencies

MehsTools is packaged with the following third parties that are grouped in the
`thirdparties` directory and automatically compiled through the cmake build
system:
* [pybind11](https://pybind11.readthedocs.io/en/stable/) v.2.2.3 which
    is a wonderfull header only library to interface python and C++ code,
* the [mapbox implementation](https://github.com/mapbox/variant) of the C++
    [variant](http://en.cppreference.com/w/cpp/utility/variant)
and [optional](http://en.cppreference.com/w/cpp/utility/optional) concepts.

# Building and installation

## Building with setuptools

If all the requirements above are accessible on your path it's possible that you
can use [pip](https://pypi.org/project/pip/)
and/or setuptools, after having set `MESHTOOLS_WITH_CGAL_DIR` environment
variable pointing to where CGAL resides (in the case you want to use it).

To have a system default installation just run in the root MeshTools directory:

```shell
pip install .
```

or if you want to have an *editable install*,
wich is fundamentally a *setuptools develop mode*, that is to say that
no files will be copied and changes to the package file
will directly directly affect the use of the module elsewhere :

```shell
pip install -e .
```

Changes to the C/C++ source files will only be available if the code is
recompiled. To do so, look for the build directory (it should be hidden in the
`build` directory on the top level of the MeshTools package repository).
From there you can use cmake to run your underlying native build tool :

```shell
cmake --build . --target install
```
or just open the Microsoft Visual Studio solution file,
or if on Linux and using make :

```shell
make install
```

## Building through CMake

Just follow the steps [here](https://cmake.org/runningcmake/), prefer
*out of source build*. The source code is the directory that you have cloned
using git, it is the top level directory where you can find a `CMakeLists.txt`
file.

Once the library is built you will have all of the python modules composing
MeshTools in the MeshTools directory. These can be imported into python
scripts and used as in the example scripts in the `tests/python` directory.
