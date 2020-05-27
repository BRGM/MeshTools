import sys

try:
    from skbuild import setup
except ImportError:
    print("scikit-build is required to build from source!", file=sys.stderr)
    print("Install it running: python -m pip install scikit-build")
    sys.exit(1)

import os
from pathlib import Path
import platform

sys.path.insert(0, Path("maintenance").as_posix())
from PackageInfo import PackageInfo

package = PackageInfo("MeshTools")

setup(
    name=package.name,
    version=package.version,
    author="various authors",
    author_email="anr-charms@brgm.fr",
    description="A python library for managing mesh information and passing it to ComPASS.",
    long_description="",
    license="GPL v.3",
    packages=["MeshTools", "MeshTools.io"],
    # FIXME: it might be more robust to use git
    #        to copy tracked files to default _skbuild directory
    cmake_install_dir=package.name,
)
